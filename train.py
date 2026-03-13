"""NBV train entrypoint."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")

from nbv_framework.scripts.train import main


if __name__ == "__main__":
    main()
