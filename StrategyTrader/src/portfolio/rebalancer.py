"""
Logica de rebalanceo de portfolio.

Soporta:
- Rebalanceo por calendario (diario, semanal, mensual)
- Rebalanceo por threshold (drift)
- Calculo de trades necesarios
- Estimacion de costos de transaccion
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .models import (
    PortfolioConfig,
    PortfolioState,
    PortfolioPosition,
    RebalanceTrade,
    RebalanceEvent,
    RebalanceFrequency,
    RebalanceReason,
)


@dataclass
class RebalanceDecision:
    """Decision de rebalanceo"""
    should_rebalance: bool
    reason: RebalanceReason
    max_drift: float = 0.0
    avg_drift: float = 0.0
    days_since_last: int = 0
    message: str = ""


class PortfolioRebalancer:
    """
    Gestor de rebalanceo de portfolio.

    Ejemplo de uso:
        rebalancer = PortfolioRebalancer(config)
        decision = rebalancer.should_rebalance(state)
        if decision.should_rebalance:
            trades = rebalancer.calculate_rebalance_trades(state, prices)
    """

    def __init__(self, config: PortfolioConfig):
        """
        Args:
            config: Configuracion del portfolio
        """
        self.config = config
        self.last_rebalance_date: Optional[datetime] = None
        self.rebalance_count: int = 0

    def should_rebalance(
        self,
        state: PortfolioState,
        current_date: Optional[datetime] = None
    ) -> RebalanceDecision:
        """
        Determina si se debe rebalancear el portfolio.

        Args:
            state: Estado actual del portfolio
            current_date: Fecha actual (default: now)

        Returns:
            RebalanceDecision con la decision y razon
        """
        if current_date is None:
            current_date = datetime.now()

        # Calcular drift
        max_drift = 0.0
        total_drift = 0.0
        num_assets = 0

        for symbol in self.config.symbols:
            current_weight = state.current_weights.get(symbol, 0.0)
            target_weight = state.target_weights.get(symbol, 0.0)
            drift = abs(current_weight - target_weight)
            max_drift = max(max_drift, drift)
            total_drift += drift
            num_assets += 1

        avg_drift = total_drift / num_assets if num_assets > 0 else 0.0

        # Calcular dias desde ultimo rebalanceo
        days_since_last = 0
        if self.last_rebalance_date:
            days_since_last = (current_date - self.last_rebalance_date).days

        # Evaluar segun frecuencia
        if self.config.rebalance_frequency == RebalanceFrequency.NEVER:
            return RebalanceDecision(
                should_rebalance=False,
                reason=RebalanceReason.SCHEDULED,
                max_drift=max_drift,
                avg_drift=avg_drift,
                days_since_last=days_since_last,
                message="Rebalancing disabled"
            )

        if self.config.rebalance_frequency == RebalanceFrequency.THRESHOLD:
            # Rebalancear si drift excede threshold
            if max_drift >= self.config.rebalance_threshold:
                return RebalanceDecision(
                    should_rebalance=True,
                    reason=RebalanceReason.THRESHOLD_BREACH,
                    max_drift=max_drift,
                    avg_drift=avg_drift,
                    days_since_last=days_since_last,
                    message=f"Max drift {max_drift:.2%} exceeds threshold {self.config.rebalance_threshold:.2%}"
                )
            return RebalanceDecision(
                should_rebalance=False,
                reason=RebalanceReason.THRESHOLD_BREACH,
                max_drift=max_drift,
                avg_drift=avg_drift,
                days_since_last=days_since_last,
                message=f"Drift {max_drift:.2%} within threshold {self.config.rebalance_threshold:.2%}"
            )

        # Rebalanceo por calendario
        should_rebalance = self._check_calendar_rebalance(current_date, days_since_last)

        if should_rebalance:
            return RebalanceDecision(
                should_rebalance=True,
                reason=RebalanceReason.SCHEDULED,
                max_drift=max_drift,
                avg_drift=avg_drift,
                days_since_last=days_since_last,
                message=f"Scheduled {self.config.rebalance_frequency.value} rebalance"
            )

        return RebalanceDecision(
            should_rebalance=False,
            reason=RebalanceReason.SCHEDULED,
            max_drift=max_drift,
            avg_drift=avg_drift,
            days_since_last=days_since_last,
            message=f"Next rebalance not due yet"
        )

    def _check_calendar_rebalance(
        self,
        current_date: datetime,
        days_since_last: int
    ) -> bool:
        """Verifica si es momento de rebalancear segun calendario"""
        # Si nunca se ha rebalanceado, hacerlo
        if self.last_rebalance_date is None:
            return True

        if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            return days_since_last >= 1

        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            # Rebalancear cada lunes (o si han pasado 7+ dias)
            is_monday = current_date.weekday() == 0
            return days_since_last >= 7 or (days_since_last >= 5 and is_monday)

        elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            # Rebalancear el primer dia del mes (o si han pasado 28+ dias)
            is_first_of_month = current_date.day == 1
            return days_since_last >= 28 or (days_since_last >= 25 and is_first_of_month)

        elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            return days_since_last >= 90

        return False

    def calculate_rebalance_trades(
        self,
        state: PortfolioState,
        prices: Dict[str, float],
        target_weights: Optional[Dict[str, float]] = None
    ) -> List[RebalanceTrade]:
        """
        Calcula los trades necesarios para rebalancear.

        Args:
            state: Estado actual del portfolio
            prices: Precios actuales por simbolo
            target_weights: Pesos objetivo (default: config.target_weights)

        Returns:
            Lista de trades a ejecutar
        """
        if target_weights is None:
            target_weights = state.target_weights

        trades = []
        total_equity = state.total_equity

        if total_equity <= 0:
            return trades

        # Calcular diferencias
        for symbol in self.config.symbols:
            current_weight = state.current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)

            weight_diff = target_weight - current_weight
            value_diff = weight_diff * total_equity

            price = prices.get(symbol, 0.0)
            if price <= 0:
                continue

            # Calcular cantidad a comprar/vender
            quantity = abs(value_diff) / price

            # Ignorar trades muy pequenos
            if abs(value_diff) < self.config.min_trade_value:
                continue

            # Determinar lado
            side = "buy" if value_diff > 0 else "sell"

            # Estimar comision
            commission = abs(value_diff) * self.config.commission

            trade = RebalanceTrade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                estimated_price=price,
                estimated_value=abs(value_diff),
                estimated_commission=commission,
                reason=RebalanceReason.SCHEDULED,
            )
            trades.append(trade)

        return trades

    def estimate_transaction_costs(
        self,
        trades: List[RebalanceTrade]
    ) -> Tuple[float, float, float]:
        """
        Estima los costos de transaccion.

        Returns:
            Tupla (total_commission, total_slippage, total_cost)
        """
        total_commission = sum(t.estimated_commission for t in trades)
        total_value = sum(t.estimated_value for t in trades)
        total_slippage = total_value * self.config.slippage
        total_cost = total_commission + total_slippage

        return total_commission, total_slippage, total_cost

    def execute_rebalance(
        self,
        state: PortfolioState,
        trades: List[RebalanceTrade],
        actual_prices: Dict[str, float],
        reason: RebalanceReason = RebalanceReason.SCHEDULED
    ) -> RebalanceEvent:
        """
        Ejecuta un rebalanceo (simula la ejecucion de trades).

        Args:
            state: Estado actual del portfolio
            trades: Trades a ejecutar
            actual_prices: Precios reales de ejecucion
            reason: Razon del rebalanceo

        Returns:
            RebalanceEvent con el resultado
        """
        event = RebalanceEvent(
            reason=reason,
            weights_before=state.current_weights.copy(),
            equity_before=state.total_equity,
        )

        new_positions = {s: p for s, p in state.positions.items()}
        total_commission = 0.0
        total_value_traded = 0.0

        for trade in trades:
            price = actual_prices.get(trade.symbol, trade.estimated_price)

            # Aplicar slippage
            slippage_mult = 1 + self.config.slippage if trade.side == "buy" else 1 - self.config.slippage
            exec_price = price * slippage_mult

            # Calcular valor real y comision
            exec_value = trade.quantity * exec_price
            commission = exec_value * self.config.commission

            total_commission += commission
            total_value_traded += exec_value

            # Actualizar posicion
            if trade.symbol in new_positions:
                pos = new_positions[trade.symbol]
                if trade.side == "buy":
                    # Agregar a la posicion
                    new_qty = pos.quantity + trade.quantity
                    new_entry = (pos.cost_basis + exec_value) / new_qty
                    new_positions[trade.symbol] = PortfolioPosition(
                        symbol=trade.symbol,
                        quantity=new_qty,
                        entry_price=new_entry,
                        current_price=price,
                        entry_date=pos.entry_date,
                    )
                else:
                    # Reducir posicion
                    new_qty = pos.quantity - trade.quantity
                    if new_qty > 0:
                        new_positions[trade.symbol] = PortfolioPosition(
                            symbol=trade.symbol,
                            quantity=new_qty,
                            entry_price=pos.entry_price,
                            current_price=price,
                            entry_date=pos.entry_date,
                        )
                    else:
                        # Posicion cerrada
                        del new_positions[trade.symbol]
            else:
                # Nueva posicion
                if trade.side == "buy":
                    new_positions[trade.symbol] = PortfolioPosition(
                        symbol=trade.symbol,
                        quantity=trade.quantity,
                        entry_price=exec_price,
                        current_price=price,
                    )

        # Actualizar state
        state.positions = new_positions
        state.cash -= total_commission  # Pagar comisiones

        # Recalcular pesos
        total_value = sum(p.value for p in new_positions.values())
        state.total_equity = state.cash + total_value
        state.invested_value = total_value
        state.update_weights()

        # Completar evento
        event.weights_after = state.current_weights.copy()
        event.equity_after = state.total_equity
        event.trades = trades
        event.total_trades = len(trades)
        event.total_value_traded = total_value_traded
        event.total_commission = total_commission
        event.success = True

        # Actualizar tracking
        self.last_rebalance_date = datetime.now()
        self.rebalance_count += 1

        return event

    def get_weight_drifts(
        self,
        state: PortfolioState
    ) -> Dict[str, float]:
        """
        Obtiene el drift de cada activo.

        Returns:
            Dict symbol -> drift (positivo = sobre-pesado, negativo = sub-pesado)
        """
        drifts = {}
        for symbol in self.config.symbols:
            current = state.current_weights.get(symbol, 0.0)
            target = state.target_weights.get(symbol, 0.0)
            drifts[symbol] = current - target
        return drifts

    def get_largest_drifts(
        self,
        state: PortfolioState,
        n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Obtiene los N activos con mayor drift absoluto.

        Returns:
            Lista de (symbol, drift) ordenada por drift absoluto
        """
        drifts = self.get_weight_drifts(state)
        sorted_drifts = sorted(
            drifts.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_drifts[:n]

    def estimate_turnover(
        self,
        state: PortfolioState,
        target_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Estima el turnover del rebalanceo.

        Returns:
            Turnover como proporcion del portfolio (0-2)
        """
        if target_weights is None:
            target_weights = state.target_weights

        turnover = 0.0
        for symbol in self.config.symbols:
            current = state.current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            turnover += abs(target - current)

        return turnover

    def reset(self) -> None:
        """Reinicia el estado del rebalancer"""
        self.last_rebalance_date = None
        self.rebalance_count = 0


class RebalancerFactory:
    """Factory para crear rebalancers preconfigurados"""

    @staticmethod
    def create_monthly(config: PortfolioConfig) -> PortfolioRebalancer:
        """Rebalancer mensual"""
        config.rebalance_frequency = RebalanceFrequency.MONTHLY
        return PortfolioRebalancer(config)

    @staticmethod
    def create_threshold_5pct(config: PortfolioConfig) -> PortfolioRebalancer:
        """Rebalancer por threshold de 5%"""
        config.rebalance_frequency = RebalanceFrequency.THRESHOLD
        config.rebalance_threshold = 0.05
        return PortfolioRebalancer(config)

    @staticmethod
    def create_quarterly(config: PortfolioConfig) -> PortfolioRebalancer:
        """Rebalancer trimestral"""
        config.rebalance_frequency = RebalanceFrequency.QUARTERLY
        return PortfolioRebalancer(config)
