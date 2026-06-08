import argparse
import sys
import os
import logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from config.config import load_config
from api.server import start_server
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default='api', choices=['api', 'train', 'test'], help='')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='')
    parser.add_argument('--checkpoint', type=str, default=None, help='')
    return parser.parse_args()
def main():
    try:
        args = parse_args()
        config = load_config(args.config)
        if args.mode == 'api':
            logger.info("API")
            start_server(config)
        elif args.mode == 'train':
            logger.info("")
            from training.trainer import Trainer
            Trainer(config).train()
        elif args.mode == 'test':
            logger.info("")
            from inference.inferencer import Inferencer
            Inferencer(config).test(args.checkpoint)
    except KeyboardInterrupt:
        logger.info("")
        sys.exit(0)
    except Exception as e:
        logger.error(f": {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == '__main__':
    main()
