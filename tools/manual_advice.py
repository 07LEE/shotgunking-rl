"""Inspect a screenshot or consented window and print advice without input."""

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from shotgun_king_rl.collection.monitor import capture_source
from shotgun_king_rl.runtime.advice import build_manual_advice, format_manual_advice
from shotgun_king_rl.runtime.perception import build_live_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--capture-backend', choices=('portal',))
    parser.add_argument('--collection', type=Path, default=ROOT / 'data/collection')
    parser.add_argument('--max-ammo', type=int, help='사용자가 확인한 최대 장전 수; 생략하면 재장전 제안 제외')
    args = parser.parse_args()
    if args.max_ammo is not None and args.max_ammo < 1:
        parser.error('--max-ammo must be positive')
    try:
        pipeline, _ = build_live_pipeline(args.collection)
        def show(image):
            if image is None:
                raise OSError('화면을 읽을 수 없습니다')
            result = pipeline.observe(image)
            print(format_manual_advice(build_manual_advice(result, max_ammo=args.max_ammo)), flush=True)
        if args.image:
            show(cv2.imread(str(args.image)))
        else:
            print('추천만 표시합니다. 게임은 직접 조작하세요. Enter: 새 화면 확인 / q: 종료', flush=True)
            with capture_source('portal', 'Shotgun King', '/usr/bin/python3') as capture:
                while True:
                    show(capture())
                    if input('\nEnter: 다시 확인 / q: 종료 > ').strip().lower() == 'q':
                        break
    except (KeyboardInterrupt, EOFError):
        print('\n추천 종료')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'추천 실패: {exc}\n')


if __name__ == '__main__':
    main()
