"""
Job Manager para tareas asincronas de Trad-loop

Maneja la ejecucion de tareas largas como:
- Backtests
- Optimizaciones
- Extraccion de datos

Cada job tiene un ID unico y se puede consultar su progreso.
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
import traceback


class JobStatus(str, Enum):
    """Estados posibles de un job"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Tipos de jobs"""
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    DATA_EXTRACTION = "data_extraction"


@dataclass
class JobProgress:
    """Progreso de un job"""
    current: int = 0
    total: int = 100
    message: str = ""

    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0
        return (self.current / self.total) * 100


@dataclass
class JobResult:
    """Resultado de un job completado"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class Job:
    """Representa un job en el sistema"""
    id: str
    type: JobType
    status: JobStatus
    params: Dict[str, Any]
    progress: JobProgress = field(default_factory=JobProgress)
    result: Optional[JobResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el job a diccionario para JSON"""
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "params": self.params,
            "progress": {
                "current": self.progress.current,
                "total": self.progress.total,
                "percentage": self.progress.percentage,
                "message": self.progress.message,
            },
            "result": asdict(self.result) if self.result else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class JobManager:
    """
    Gestor de jobs asincrono.

    Uso:
        manager = JobManager()

        # Crear un job
        job_id = await manager.create_job(
            job_type=JobType.BACKTEST,
            params={"strategy": "EMA", "symbol": "BTC/USDT"},
            executor=run_backtest_func
        )

        # Consultar estado
        job = manager.get_job(job_id)
        print(job.status, job.progress.percentage)

        # Cancelar
        await manager.cancel_job(job_id)
    """

    def __init__(self, max_workers: int = 4):
        self._jobs: Dict[str, Job] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()

    async def create_job(
        self,
        job_type: JobType,
        params: Dict[str, Any],
        executor: Callable,
        run_sync: bool = True
    ) -> str:
        """
        Crea y ejecuta un nuevo job.

        Args:
            job_type: Tipo de job (BACKTEST, OPTIMIZATION, etc)
            params: Parametros para el job
            executor: Funcion que ejecuta el job
                     Debe aceptar (params, progress_callback) y retornar resultado
            run_sync: Si True, ejecuta en ThreadPool (para codigo sincrono)
                     Si False, ejecuta como coroutine async

        Returns:
            ID del job creado
        """
        job_id = str(uuid.uuid4())[:8]

        job = Job(
            id=job_id,
            type=job_type,
            status=JobStatus.QUEUED,
            params=params,
        )

        async with self._lock:
            self._jobs[job_id] = job

        # Crear task para ejecutar el job
        task = asyncio.create_task(
            self._run_job(job, executor, run_sync)
        )
        self._tasks[job_id] = task

        return job_id

    async def _run_job(
        self,
        job: Job,
        executor: Callable,
        run_sync: bool
    ) -> None:
        """Ejecuta un job"""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()

        def progress_callback(current: int, total: int, message: str = ""):
            """Callback para actualizar progreso"""
            job.progress.current = current
            job.progress.total = total
            job.progress.message = message

        try:
            if run_sync:
                # Ejecutar funcion sincrona en thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: executor(job.params, progress_callback)
                )
            else:
                # Ejecutar coroutine async
                result = await executor(job.params, progress_callback)

            job.result = JobResult(success=True, data=result)
            job.status = JobStatus.COMPLETED

        except asyncio.CancelledError:
            job.result = JobResult(success=False, error="Job cancelled")
            job.status = JobStatus.CANCELLED
            raise

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            job.result = JobResult(success=False, error=error_msg)
            job.status = JobStatus.FAILED

        finally:
            job.completed_at = datetime.now()
            job.progress.current = job.progress.total  # Marcar como 100%

    def get_job(self, job_id: str) -> Optional[Job]:
        """Obtiene un job por ID"""
        return self._jobs.get(job_id)

    def get_all_jobs(self, job_type: Optional[JobType] = None) -> List[Job]:
        """Obtiene todos los jobs, opcionalmente filtrados por tipo"""
        jobs = list(self._jobs.values())
        if job_type:
            jobs = [j for j in jobs if j.type == job_type]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancela un job en ejecucion"""
        if job_id not in self._tasks:
            return False

        task = self._tasks[job_id]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        return False

    def clear_completed(self, older_than_hours: int = 24) -> int:
        """Limpia jobs completados mas antiguos que X horas"""
        cutoff = datetime.now()
        count = 0

        to_remove = []
        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at:
                    age_hours = (cutoff - job.completed_at).total_seconds() / 3600
                    if age_hours > older_than_hours:
                        to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]
            if job_id in self._tasks:
                del self._tasks[job_id]
            count += 1

        return count


# Instancia global del job manager
job_manager = JobManager()


def get_job_manager() -> JobManager:
    """Obtiene la instancia global del job manager"""
    return job_manager
