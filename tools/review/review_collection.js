const token = document.querySelector('meta[name=review-token]').content, classes = ['empty', 'player_king', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'white_king'];
const $ = id => document.getElementById(id); let items = [], data, revision, selected = [0, 0], dirty = false, currentSession = '';
function remember() { try { localStorage.setItem('piece-review-selection', JSON.stringify({ session: currentSession, id: currentId })) } catch (e) { } }
function recalled() { try { return JSON.parse(localStorage.getItem('piece-review-selection')) || {} } catch (e) { return {} } }
function sessionItems() { return items.filter(item => item.session === currentSession) }
for (const c of classes) { const o = document.createElement('option'); o.value = c; o.textContent = c; $('label').append(o) }
async function api(url, options) { const r = await fetch(url, options); const a = await r.json(); if (!r.ok) throw Error(a.error || r.statusText); return a }
function status(s) { $('status').textContent = s }
function exclude(r, c) { return data.exclude_cells.some(x => x[0] === r && x[1] === c) }
function draw() {
    const grid = $('grid');
    grid.replaceChildren();
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const button = document.createElement('button');
            button.className = 'cell' + (exclude(r, c) ? ' excluded' : '') + (selected[0] === r && selected[1] === c ? ' selected' : '');
            const square = String.fromCharCode(97 + c) + (8 - r);
            button.title = square + ': ' + data.board[r][c];
            button.setAttribute('aria-label', button.title);
            const text = document.createElement('span');
            text.className = data.board[r][c] === 'empty' ? 'empty-label' : 'piece-label';
            text.textContent = data.board[r][c].replace('player_king', 'Player king').replace('white_king', 'White king');
            button.append(text);
            button.onclick = () => { selected = [r, c]; draw(); };
            grid.append(button);
        }
    }
    const [r, c] = selected;
    $('label').value = data.board[r][c];
    $('excluded').checked = exclude(r, c);
    $('cellInfo').textContent = String.fromCharCode(97 + c) + (8 - r) + ': ' + data.board[r][c];
}
function labelView() {
    $('grid').classList.toggle('show-labels', $('showLabels').checked);
}
$('showLabels').onchange = labelView;
labelView();
function cropValue() { const parts = $('crop').value.split(',').map(x => x.trim()); if (parts.length !== 4 || parts.some(x => !/^\d+$/.test(x))) throw Error('Enter four nonnegative integer crop coordinates'); return parts.map(Number) }
function preview() { const crop = cropValue(); $('image').src = '/api/board?id=' + encodeURIComponent($('items').value) + '&crop=' + encodeURIComponent(JSON.stringify(crop)); if (data) draw(); }
async function load() { try { const a = await api('/api/item?id=' + encodeURIComponent($('items').value)); data = a.data; revision = a.revision; data.exclude_cells ??= []; selected = [0, 0]; $('floor').value = data.floor ?? ''; $('crop').value = data.game_crop.join(', '); $('split').textContent = 'Session split: ' + data.split; preview(); draw(); dirty = false; status(data.confirmed ? 'Previously approved. Editing requires saving again.' : 'Draft: review all labels before approval.') } catch (e) { status(e.message) } }
$('image').onerror = () => status('Board preview failed. Check crop coordinates.');
$('items').onchange = () => { if (dirty && !confirm('Discard unsaved edits?')) { $('items').value = currentId; return } currentId = $('items').value; remember(); renderOptions(); load() }; let currentId = '';
$('sessions').onchange = () => { if (dirty && !confirm('Discard unsaved edits?')) { $('sessions').value = currentSession; return } currentSession = $('sessions').value; const list = sessionItems(); currentId = list.find(x => !x.confirmed)?.id || list[0].id; remember(); renderOptions(); load() };
$('label').onchange = () => { const [r, c] = selected; data.board[r][c] = $('label').value; dirty = true; draw() };
$('excluded').onchange = () => { const [r, c] = selected; data.exclude_cells = data.exclude_cells.filter(x => x[0] !== r || x[1] !== c); if ($('excluded').checked) data.exclude_cells.push([r, c]); dirty = true; draw() };
for (const id of ['floor', 'crop']) $(id).oninput = () => { dirty = true };
$('preview').onclick = () => { try { preview() } catch (e) { status(e.message) } };
async function save(confirmed) {
    const buttons = [$('draft'), $('approve')];
    buttons.forEach(button => { button.disabled = true; });
    try {
        const raw = $('floor').value;
        const floor = raw === '' ? null : Number(raw);
        if (floor !== null && (!Number.isInteger(floor) || floor < 1)) throw Error('Floor must be a positive integer');
        const result = await api('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Review-Token': token },
            body: JSON.stringify({ id: currentId, revision, board: data.board, exclude_cells: data.exclude_cells, game_crop: cropValue(), floor, split: data.split, confirmed })
        });
        revision = result.revision;
        data.confirmed = confirmed;
        dirty = false;
        items.find(item => item.id === currentId).confirmed = confirmed;
        renderOptions();
        if (!confirmed) {
            status('Draft saved; excluded from approved datasets.');
            return;
        }
        const list = sessionItems();
        const index = list.findIndex(item => item.id === currentId);
        const next = [...list.slice(index + 1), ...list.slice(0, index)].find(item => !item.confirmed);
        if (next) {
            currentId = next.id;
            remember();
            renderOptions();
            await load();
        } else {
            status('Session review complete — all ' + list.length + ' screens approved.');
        }
    } catch (error) {
        status(error.message);
    } finally {
        buttons.forEach(button => { button.disabled = false; });
    }
}
$('draft').onclick = () => save(false); $('approve').onclick = () => save(true);
function renderOptions() { const list = sessionItems(); $('items').replaceChildren(); for (const item of list) { const o = document.createElement('option'); o.value = item.id; o.textContent = (item.confirmed ? '✓ ' : '○ ') + item.id.split('/').pop(); $('items').append(o) } $('items').value = currentId; const index = list.findIndex(x => x.id === currentId); $('previous').disabled = index <= 0; $('next').disabled = index >= list.length - 1; $('progress').textContent = list.filter(x => x.confirmed).length + ' / ' + list.length + ' approved · screenshot ' + (index + 1) + ' / ' + list.length }
function navigate(delta) { const list = sessionItems(); const index = list.findIndex(x => x.id === currentId); if (index + delta < 0 || index + delta >= list.length) return; if (dirty && !confirm('Discard unsaved edits?')) return; currentId = list[index + delta].id; remember(); renderOptions(); load() }
$('previous').onclick = () => navigate(-1); $('next').onclick = () => navigate(1);
window.onbeforeunload = e => { if (dirty) { e.preventDefault(); e.returnValue = '' } };
(async () => { try { items = await api('/api/items'); if (!items.length) { status('No annotations. Run with --session to prepare inbox images.'); return } const sessions = [...new Set(items.map(x => x.session))]; const saved = recalled(); currentSession = sessions.includes(saved.session) ? saved.session : sessions[0]; for (const session of sessions) { const o = document.createElement('option'); o.value = session; o.textContent = session; $('sessions').append(o) } $('sessions').value = currentSession; const list = sessionItems(); currentId = list.some(x => x.id === saved.id) ? saved.id : (list.find(x => !x.confirmed)?.id || list[0].id); remember(); renderOptions(); await load() } catch (e) { status(e.message) } })();
