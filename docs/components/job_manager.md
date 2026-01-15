# Job Manager

The Job Manager is an asynchronous task execution system that handles long-running operations such as backtests, optimizations, and data extraction in the background.

## Overview

The Job Manager provides:
- Background execution of CPU-intensive tasks
- Progress tracking and status updates
- Task cancellation support
- Thread pool for synchronous code execution
- Automatic cleanup of old completed jobs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Job Manager                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   REST API  │───▶│  JobManager │───▶│ ThreadPool  │         │
│  │  Endpoint   │    │             │    │  Executor   │         │
│  └─────────────┘    └──────┬──────┘    └─────────────┘         │
│                            │                                     │
│                            ▼                                     │
│                     ┌─────────────┐                             │
│                     │    Jobs     │                             │
│                     │  Dictionary │                             │
│                     └─────────────┘                             │
│                            │                                     │
│            ┌───────────────┼───────────────┐                    │
│            ▼               ▼               ▼                    │
│      ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│      │ Backtest │    │Optimizat.│    │  Data    │              │
│      │   Job    │    │   Job    │    │Extraction│              │
│      └──────────┘    └──────────┘    └──────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Job Types

| Type | Description |
|------|-------------|
| `BACKTEST` | Execute strategy backtests |
| `OPTIMIZATION` | Run parameter optimization |
| `DATA_EXTRACTION` | Extract historical market data |

## Job States

| State | Description |
|-------|-------------|
| `QUEUED` | Job created and waiting to start |
| `RUNNING` | Job is currently executing |
| `COMPLETED` | Job finished successfully |
| `FAILED` | Job encountered an error |
| `CANCELLED` | Job was cancelled by user |

## Core Classes

### JobProgress

Tracks the progress of a running job.

```python
@dataclass
class JobProgress:
    current: int = 0      # Current progress step
    total: int = 100      # Total steps
    message: str = ""     # Progress message

    @property
    def percentage(self) -> float:
        """Returns progress as percentage (0-100)"""
        if self.total == 0:
            return 0
        return (self.current / self.total) * 100
```

### JobResult

Contains the result of a completed job.

```python
@dataclass
class JobResult:
    success: bool                        # Whether job succeeded
    data: Optional[Dict[str, Any]]       # Result data (if successful)
    error: Optional[str]                 # Error message (if failed)
```

### Job

Represents a job in the system.

```python
@dataclass
class Job:
    id: str                              # Unique job identifier
    type: JobType                        # Type of job
    status: JobStatus                    # Current status
    params: Dict[str, Any]               # Job parameters
    progress: JobProgress                # Progress tracking
    result: Optional[JobResult]          # Result (when completed)
    created_at: datetime                 # Creation timestamp
    started_at: Optional[datetime]       # Start timestamp
    completed_at: Optional[datetime]     # Completion timestamp
```

### JobManager

Main class for managing jobs.

```python
class JobManager:
    def __init__(self, max_workers: int = 4):
        """Initialize with thread pool size"""

    async def create_job(
        self,
        job_type: JobType,
        params: Dict[str, Any],
        executor: Callable,
        run_sync: bool = True
    ) -> str:
        """Create and start a new job"""

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""

    def get_all_jobs(self, job_type: Optional[JobType] = None) -> List[Job]:
        """Get all jobs, optionally filtered by type"""

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""

    def clear_completed(self, older_than_hours: int = 24) -> int:
        """Remove old completed jobs"""
```

## Usage Examples

### Creating a Backtest Job

```python
from src.job_manager import get_job_manager, JobType

async def run_backtest(params: dict, progress_callback):
    """Backtest executor function"""
    strategy_class = params['strategy_class']
    data = params['data']

    # Update progress
    progress_callback(0, 100, "Initializing strategy...")

    strategy = strategy_class(**params.get('strategy_params', {}))
    strategy.load_data(data)

    progress_callback(25, 100, "Running backtest...")
    trades = strategy.backtest()

    progress_callback(75, 100, "Calculating metrics...")
    metrics = strategy.get_performance_metrics()

    progress_callback(100, 100, "Complete")

    return {
        'trades': trades.to_dict(),
        'metrics': metrics,
        'equity_curve': strategy.equity_curve.to_dict()
    }

# Create job
manager = get_job_manager()
job_id = await manager.create_job(
    job_type=JobType.BACKTEST,
    params={
        'strategy_class': MyStrategy,
        'data': ohlcv_data,
        'strategy_params': {'fast_period': 12, 'slow_period': 26}
    },
    executor=run_backtest
)

print(f"Job started with ID: {job_id}")
```

### Checking Job Status

```python
# Get job status
job = manager.get_job(job_id)

if job:
    print(f"Status: {job.status.value}")
    print(f"Progress: {job.progress.percentage:.1f}%")
    print(f"Message: {job.progress.message}")

    if job.status == JobStatus.COMPLETED:
        print(f"Result: {job.result.data}")
    elif job.status == JobStatus.FAILED:
        print(f"Error: {job.result.error}")
```

### Cancelling a Job

```python
cancelled = await manager.cancel_job(job_id)
if cancelled:
    print("Job cancelled successfully")
else:
    print("Could not cancel job (may be already completed)")
```

### Creating an Optimization Job

```python
async def run_optimization(params: dict, progress_callback):
    """Optimization executor function"""
    optimizer = StrategyOptimizer(
        strategy_class=params['strategy_class'],
        data=params['data'],
        initial_capital=params.get('initial_capital', 10000)
    )

    for name, config in params['parameters'].items():
        optimizer.add_parameter(name, **config)

    total_combinations = optimizer.get_total_combinations()

    def update_progress(current):
        progress_callback(current, total_combinations, f"Testing combination {current}/{total_combinations}")

    results = optimizer.grid_optimize(
        objective=params.get('objective', 'sharpe_ratio'),
        progress_callback=update_progress
    )

    return {
        'best_params': results.best_params,
        'best_value': results.best_value,
        'all_results': [r.to_dict() for r in results.all_results]
    }

job_id = await manager.create_job(
    job_type=JobType.OPTIMIZATION,
    params={
        'strategy_class': MyStrategy,
        'data': data,
        'parameters': {
            'fast_period': {'type': 'int', 'min': 5, 'max': 20, 'step': 5},
            'slow_period': {'type': 'int', 'min': 20, 'max': 50, 'step': 10}
        },
        'objective': 'sharpe_ratio'
    },
    executor=run_optimization
)
```

### Async Executor

For async functions, set `run_sync=False`:

```python
async def async_executor(params: dict, progress_callback):
    """Async executor example"""
    async with aiohttp.ClientSession() as session:
        # Async operations...
        progress_callback(50, 100, "Fetching data...")
        async with session.get(url) as response:
            data = await response.json()
        progress_callback(100, 100, "Complete")
        return data

job_id = await manager.create_job(
    job_type=JobType.DATA_EXTRACTION,
    params={'url': 'https://api.example.com/data'},
    executor=async_executor,
    run_sync=False  # Execute as coroutine
)
```

## REST API Integration

The Job Manager integrates with the REST API for web-based job management.

### Create Job Endpoint

```
POST /api/v1/jobs/backtest
```

Request body:
```json
{
    "strategy": "EMA_Crossover",
    "symbol": "BTC/USDT",
    "exchange": "binance",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 10000,
    "strategy_params": {
        "fast_period": 12,
        "slow_period": 26
    }
}
```

Response:
```json
{
    "job_id": "a1b2c3d4",
    "status": "queued",
    "message": "Backtest job created"
}
```

### Get Job Status Endpoint

```
GET /api/v1/jobs/{job_id}
```

Response:
```json
{
    "id": "a1b2c3d4",
    "type": "backtest",
    "status": "running",
    "progress": {
        "current": 75,
        "total": 100,
        "percentage": 75.0,
        "message": "Calculating metrics..."
    },
    "created_at": "2024-01-15T10:30:00Z",
    "started_at": "2024-01-15T10:30:01Z",
    "completed_at": null
}
```

### List All Jobs Endpoint

```
GET /api/v1/jobs?type=backtest
```

Response:
```json
{
    "jobs": [
        {
            "id": "a1b2c3d4",
            "type": "backtest",
            "status": "completed",
            "progress": {"current": 100, "total": 100, "percentage": 100.0}
        },
        {
            "id": "e5f6g7h8",
            "type": "backtest",
            "status": "running",
            "progress": {"current": 50, "total": 100, "percentage": 50.0}
        }
    ]
}
```

### Cancel Job Endpoint

```
POST /api/v1/jobs/{job_id}/cancel
```

Response:
```json
{
    "success": true,
    "message": "Job cancelled"
}
```

## Global Instance

The Job Manager provides a global singleton instance:

```python
from src.job_manager import get_job_manager

# Get the global instance
manager = get_job_manager()

# The instance is shared across the application
# This ensures all jobs are managed centrally
```

## Configuration

The Job Manager can be configured with:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_workers` | 4 | Maximum concurrent thread pool workers |

```python
# Custom configuration
from src.job_manager import JobManager

custom_manager = JobManager(max_workers=8)
```

## Best Practices

1. **Progress Updates**: Provide frequent progress updates for long-running jobs
2. **Error Handling**: Always handle exceptions in executor functions
3. **Cleanup**: Periodically call `clear_completed()` to free memory
4. **Cancellation**: Check for cancellation in long loops:
   ```python
   import asyncio

   async def executor(params, progress_callback):
       for i in range(100):
           # Allow cancellation check
           await asyncio.sleep(0)
           # Do work...
           progress_callback(i, 100, f"Step {i}")
   ```
5. **Resource Limits**: Configure `max_workers` based on available CPU cores

## Related Modules

- [API Reference](api_reference.md) - REST API endpoints
- [Optimizers](optimizers.md) - Optimization algorithms
- [Performance](../user_guide.md#performance-analysis) - Performance analysis
