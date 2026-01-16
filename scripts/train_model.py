"""
Скрипт для навчання моделі на training датасеті
Wrapper до runtime.bootstrap.bootstrap()
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from runtime.bootstrap import bootstrap


def main():
    """
    Wrapper до runtime.bootstrap.bootstrap()
    Зберігає backwards compatibility зі старими параметрами CLI
    """
    parser = argparse.ArgumentParser(
        description="Навчання TRM моделі - wrapper до runtime.bootstrap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  # Навчання з config.yaml:
  python scripts/train_model.py --config config/config.yaml
  
  # Продовження навчання:
  python scripts/train_model.py --mode resume
  
  # Service режим:
  python scripts/train_model.py --mode service
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Шлях до config.yaml (за замовчуванням: config/config.yaml)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="new",
        choices=["new", "resume", "service"],
        help="Режим навчання (new, resume, service)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Шлях до checkpoint для продовження навчання"
    )
    
    args = parser.parse_args()
    
    # Викликати bootstrap
    bootstrap(
        config_path=args.config,
        mode=args.mode,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
