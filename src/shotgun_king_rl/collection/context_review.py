"""Export failed side cells and their surroundings without changing labels."""

import argparse
import base64
import hashlib
from html import escape
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..collection_layout import annotation_paths, validate_session_id
from ..perception import BoardContextReader, OverlayScreenReader, TemplatePieceReader, ScreenKind
from ..perception.frame import PreparedFrame, read_frame
from ..vision.layout import normalize_game_screen


REASONS = {'indicator_occlusion': '색 표시가 칸을 가려 바닥·기물 확인 불가', 'unmatched_cell': '검수 참조와 일치하지 않음',
           'conflicting_cell': '소속 참조가 서로 충돌함',
           'identity_conflict': '조종 왕 식별과 소속 근거가 충돌함',
           'references_unavailable': '검수된 소속 참조 없음'}
PIECES = {0: '빈칸', 1: '조종 왕', 2: '폰', 3: '나이트', 4: '비숍', 5: '룩',
          6: '왕 계열 미확정', 7: '특수 나이트', 8: '퀸', 9: '백왕'}


def source_cell_box(crop, row, col):
    left, top, right, bottom = crop
    sx, sy = (right - left) / 1280, (bottom - top) / 720
    return [math.floor(left + (383 + col * 64) * sx),
            math.floor(top + (119 + row * 64) * sy),
            math.ceil(left + (383 + (col + 1) * 64) * sx),
            math.ceil(top + (119 + (row + 1) * 64) * sy)]


def _embedded(image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise OSError('Could not encode review image')
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def _save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not save review image: {path}')


def export_context_review(collection, session, output):
    validate_session_id(session)
    root = Path(collection).resolve()
    output = Path(output).resolve()
    paths = list(annotation_paths(root, [session]))
    if not paths:
        raise ValueError('No annotations found for this session')
    if output.exists():
        raise ValueError('Output exists; choose a new review directory')
    originals = root / 'originals'
    # Refuse output paths inside source data directories.
    if any(output.is_relative_to(root / name) for name in ('annotations', 'originals', 'inbox')):
        raise ValueError('Review output must be separate from source data')
    context = BoardContextReader.from_collection(root)
    pieces = TemplatePieceReader.from_collection(root)
    screen_reader = OverlayScreenReader()
    output.mkdir(parents=True)
    assets = output / 'images'
    assets.mkdir()
    records, sections, tiles = [], [], []
    for index, path in enumerate(sorted(paths), 1):
        annotation_bytes = path.read_bytes()
        data = json.loads(annotation_bytes)
        original = (path.parent / data['image']).resolve()
        if original.parent != originals:
            raise ValueError(f'Invalid original path: {path.name}')
        original_bytes = original.read_bytes()
        image = cv2.imdecode(np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_COLOR)
        frame = PreparedFrame(normalize_game_screen(image, data['game_crop']), normalized=True)
        screen = read_frame(screen_reader, frame)
        record = {'index': index, 'annotation': path.name, 'original': str(original),
                  'annotation_sha256': hashlib.sha256(annotation_bytes).hexdigest(),
                  'original_sha256': hashlib.sha256(original_bytes).hexdigest(),
                  'screen_kind': screen.value.value if screen.is_accepted else None,
                  'cells': []}
        records.append(record)
        if not screen.is_accepted or screen.value is not ScreenKind.GAMEPLAY:
            record['skipped_reason'] = screen.reason or f'screen:{record["screen_kind"]}'
            continue
        board_prediction = read_frame(pieces, frame)
        board = board_prediction.value if board_prediction.is_accepted else None
        record['piece_board_accepted'] = board_prediction.is_accepted
        record['piece_board_reason'] = board_prediction.reason
        overview = image.copy()
        cells_html = []
        side_cells = context.read_side_cells(frame.board_image, board=board)
        for row in range(8):
            for col in range(8):
                patch = frame.board_image[row*65:(row+1)*65, col*65:(col+1)*65]
                prediction = side_cells[row][col]
                if prediction.is_accepted:
                    continue
                square = f'{chr(97+col)}{8-row}'
                key = f'{index:02d}-{square}'
                source_box = source_cell_box(data['game_crop'], row, col)
                item = {'square': square, 'row': row, 'col': col, 'source_box': source_box,
                        'reason': prediction.reason, 'reader_source': prediction.source,
                        'piece_prediction': int(board[row, col]) if board is not None else None,
                        'draft_piece_label': data['board'][row][col],
                        'patch': f'images/{key}-cell.png', 'neighborhood': f'images/{key}-neighbors.png'}
                record['cells'].append(item)
                cell_image = cv2.resize(patch, (260, 260), interpolation=cv2.INTER_NEAREST)
                tiles.append((index, square, cell_image))
                r0, r1 = max(0, row-1)*65, min(8, row+2)*65
                c0, c1 = max(0, col-1)*65, min(8, col+2)*65
                neighbors = frame.board_image[r0:r1, c0:c1].copy()
                cv2.rectangle(neighbors, (col*65-c0, row*65-r0), (col*65-c0+64, row*65-r0+64), (0, 0, 255), 1)
                neighbors = cv2.resize(neighbors, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                _save_image(output / item['patch'], cell_image)
                _save_image(output / item['neighborhood'], neighbors)
                left, top, right, bottom = source_box
                cv2.rectangle(overview, (left, top), (right-1, bottom-1), (0, 0, 255), 2)
                cv2.putText(overview, square, (left+2, top+14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
                reason = REASONS.get(prediction.reason, prediction.reason)
                piece = PIECES.get(item['piece_prediction'], '미수락')
                cells_html.append(f'<article id="{key}"><h3>{square} · {escape(reason)}</h3>'
                                  f'<p>기물 자동 판독: {escape(piece)} · 원본 좌표: {source_box}</p>'
                                  f'<div class="pair"><figure><img src="{_embedded(cell_image)}" alt="{square} 확대"><figcaption>해당 셀 확대</figcaption></figure>'
                                  f'<figure><img src="{_embedded(neighbors)}" alt="{square} 주변"><figcaption>주변 칸 · 빨간 테두리가 대상</figcaption></figure></div></article>')
        record['moat_reason'] = context.moat_reader.read_board(frame.board_image).reason
        if record['cells']:
            _save_image(assets / f'{index:02d}-overview.png', overview)
            sections.append(f'<section id="frame-{index}"><h2>{index}번 · 미확정 {len(record["cells"])}칸</h2>'
                            f'<p class="filename">{escape(path.name)}</p>'
                            f'<img class="overview" src="{_embedded(overview)}" alt="원본 위치 표시">'
                            + ''.join(cells_html) + '</section>')
        if path.read_bytes() != annotation_bytes or original.read_bytes() != original_bytes:
            raise RuntimeError('Source data changed during review export')
    total = sum(len(record['cells']) for record in records)
    if tiles:
        sheet = np.full((math.ceil(len(tiles)/6)*240, 1260, 3), (42, 35, 32), np.uint8)
        for position, (index, square, cell_image) in enumerate(tiles):
            x, y = (position % 6)*210, (position // 6)*240
            sheet[y+30:y+230, x:x+200] = cv2.resize(cell_image, (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.putText(sheet, f'{index:02d} / {square}', (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        _save_image(output / 'all-cells.png', sheet)
    summary = {'session': session, 'frames': len(records), 'frames_with_failed_cells': sum(bool(r['cells']) for r in records),
               'failed_cells': total, 'annotations_changed': False, 'approvals_changed': False,
               'context_metadata': context.metadata(), 'records': records}
    (output / 'review.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    navigation = ' · '.join(f'<a href="#frame-{r["index"]}">{r["index"]}번 ({len(r["cells"])}칸)</a>' for r in records if r['cells'])
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>미확정 셀 검수</title><style>body{max-width:1100px;margin:24px auto;padding:0 18px;background:#181a20;color:#eee;font:16px sans-serif;line-height:1.6}a{color:#8bc8ff}.overview{width:100%;max-width:900px}section{margin-top:40px}article{padding:14px;background:#242832;margin:16px 0}.pair{display:flex;gap:20px;flex-wrap:wrap}figure{margin:0}figure img{max-width:100%;image-rendering:pixelated}.filename{overflow-wrap:anywhere;color:#bbb}nav{position:sticky;top:0;background:#181a20;padding:10px}</style>'
            f'<h1>미확정 셀 검수 · {escape(session)}</h1><p>{len(records)}장 중 {summary["frames_with_failed_cells"]}장 · 미확정 {total}칸</p>'
            '<p>기물 이름은 자동 판독 제안이며 정답이 아닙니다. 실제 기물·인접 그림 침범·가림을 이미지로 확인하세요. 라벨과 승인은 변경하지 않았습니다.</p>'
            f'<nav>{navigation}</nav>' + ''.join(sections) + '</html>')
    (output / 'index.html').write_text(html)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=Path(__file__).resolve().parents[3] / 'data/collection')
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = export_context_review(args.collection, args.session, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'검수 자료 생성 실패: {exc}\n')
    print(f'{report["frames_with_failed_cells"]}장 / {report["failed_cells"]}칸: {args.output / "index.html"}')


if __name__ == '__main__':
    main()
