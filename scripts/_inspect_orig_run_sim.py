import inspect
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import predictive_service_orig
print(inspect.getsource(predictive_service_orig.run_simulation))
