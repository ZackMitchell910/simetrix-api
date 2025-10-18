from pathlib import Path

path = Path('src/services/simulation.py')
text = path.read_text()
needle_start = 'async def run_simulation'
needle_end = '\nasync def list_models'
start = text.index(needle_start)
end = text.index(needle_end, start)
new_block = """async def run_simulation(run_id: str, req: SimRequest, redis: Redis) -> None:\n    from src import predictive_api\n    predictive_api.REDIS = redis\n    if not hasattr(predictive_api, '_ensure_run'):\n        predictive_api._ensure_run = _ensure_run\n    if not hasattr(predictive_api, '_update_run_state'):\n        predictive_api._update_run_state = _update_run_state\n    await predictive_api.run_simulation(run_id, req, redis)\n\n"""
path.write_text(text[:start] + new_block + text[end:])
