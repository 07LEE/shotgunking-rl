const ammoInputs = { loaded: 'ammo_loaded', reserve: 'ammo_reserve' };
const combatFields = ['attack', 'range_min', 'range_max', 'spread_degrees', 'penetration_percent', 'knockback_percent', 'grenades_remaining', 'grenade_damage', 'melee_damage'];
const reviewInputs = { screen_kind: 'screenKindReviewed', ammo: 'ammoReviewed', combat_stats: 'combatReviewed' };
const independentReviewInputs = { soul: 'soulReviewed' };
const soulPieces = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'white_king', 'special_knight'];
const token = document.querySelector('meta[name=review-token]').content, classes = ['ambiguous_royal', 'empty', 'player_king', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'white_king', 'special_knight'];
const $ = id => document.getElementById(id); let items = [], data, revision, selected = [0, 0], dirty = false, currentSession = '', currentReviewTab = 'pieces', pendingPieceApprovalId = null;
function remember() { try { localStorage.setItem('piece-review-selection', JSON.stringify({ session: currentSession, id: currentId })) } catch (e) { } }
function recalled() { try { return JSON.parse(localStorage.getItem('piece-review-selection')) || {} } catch (e) { return {} } }
function sessionItems() { return items.filter(item => item.session === currentSession) }
function nonGameplay(kind) { return Boolean(kind && kind !== 'gameplay') }
function pieceApplicable(item) { return !nonGameplay(item.screen_kind) }
function pageSections(item) { return nonGameplay(item.screen_kind) ? ['screen_kind'] : Object.keys(reviewInputs) }
function pageReviewCount(item) { return pageSections(item).filter(section => item.reviews?.[section] === true).length }
function pageComplete(item) { return pageReviewCount(item) === pageSections(item).length }
function soulReviewPending(item) { return item.screen_kind === 'gameplay' && item.reviews?.soul !== true }
function pageWorkflowComplete(item) { return pageComplete(item) && !soulReviewPending(item) }
function needsReview(item) { return currentReviewTab === 'pieces' ? pieceApplicable(item) && !item.confirmed : !pageWorkflowComplete(item) }
function firstPending(list) { return list.find(needsReview) || list[0] }
function otherSessionPending() {
    const sessions = [...new Set(items.map(item => item.session))];
    const index = sessions.indexOf(currentSession);
    const ordered = [...sessions.slice(index + 1), ...sessions.slice(0, index)];
    for (const session of ordered) {
        const pending = items.find(item => item.session === session && needsReview(item));
        if (pending) return pending;
    }
    return null;
}
function nextPending() {
    const list = sessionItems();
    const index = list.findIndex(item => item.id === currentId);
    return list.slice(index + 1).find(needsReview) || otherSessionPending();
}
async function moveTo(item) {
    if (item.id !== currentId) pendingPieceApprovalId = null;
    if (item.session !== currentSession) {
        currentSession = item.session;
        $('sessions').value = currentSession;
    }
    currentId = item.id;
    remember();
    renderOptions();
    await load();
}
function reviewMarker(item) { if (currentReviewTab === 'pieces') return pieceApplicable(item) ? item.confirmed ? '✓' : '○' : '—'; if (pageWorkflowComplete(item)) return '✓'; return pageReviewCount(item) > 0 || item.reviews?.soul === true ? '◐' : '○' }
function showReviewStatus() { if (!data) return; const item = items.find(candidate => candidate.id === currentId); const approvedAt = item.piece_review?.reviewed_at; const pieceMessage = !pieceApplicable(item) ? 'Piece review does not apply to this screen type.' : !data.confirmed ? 'Piece labels are still a draft.' : approvedAt ? 'Piece labels approved · ' + new Date(approvedAt).toLocaleString() : 'Piece labels approved · approval time unavailable.'; const pageMessage = !pageComplete(item) ? 'Page information review is incomplete.' : soulReviewPending(item) ? 'Page information reviewed · harvested souls still need review.' : 'Page information and harvested souls reviewed.'; status(currentReviewTab === 'pieces' ? pieceMessage : pageMessage) }
for (const c of classes) { const o = document.createElement('option'); o.value = c; o.textContent = c === 'ambiguous_royal' ? 'Queen / white king (unresolved)' : c; $('label').append(o) }
async function api(url, options) { const r = await fetch(url, options); const a = await r.json(); if (!r.ok) throw Error(a.error || r.statusText); return a }
function status(s, error = false) {
    $('status').textContent = s;
    $('status').classList.toggle('error', error);
    if (error) $('status').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}
function exclude(r, c) { return data.exclude_cells.some(x => x[0] === r && x[1] === c) }
function draw() {
    const grid = $('grid');
    grid.replaceChildren();
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const button = document.createElement('button');
            button.className = 'cell' + (exclude(r, c) ? ' excluded' : '') + (selected[0] === r && selected[1] === c ? ' selected' : '');
            const square = String.fromCharCode(97 + c) + (8 - r);
            button.classList.add(data.terrain[r][c] === 'moat' ? 'moat' : 'plain');
            button.title = square + ': ' + data.board[r][c] + ' · ' + (data.piece_sides[r][c] ?? 'no piece') + ' · ' + data.terrain[r][c];
            button.setAttribute('aria-label', button.title);
            const text = document.createElement('span');
            text.className = data.board[r][c] === 'empty' ? 'empty-label' : 'piece-label';
            text.textContent = (data.piece_sides[r][c] === 'ally' && data.board[r][c] !== 'player_king' ? 'Ally ' : '') + data.board[r][c].replace('ambiguous_royal', 'Queen / king?').replace('player_king', 'Player king').replace('white_king', 'White king').replace('special_knight', 'Special N');
            button.append(text);
            button.onclick = () => { selected = [r, c]; draw(); };
            grid.append(button);
        }
    }
    const [r, c] = selected;
    $('label').value = data.board[r][c];
    $('pieceSide').value = data.piece_sides[r][c] ?? '';
    $('pieceSide').disabled = ['empty', 'player_king'].includes(data.board[r][c]);
    $('excluded').checked = exclude(r, c);
    $('locked').checked = data.screen_state.locked_cells.some(x => x[0] === r && x[1] === c);
    $('cellInfo').textContent = String.fromCharCode(97 + c) + (8 - r) + ': ' + data.board[r][c] + ' · ' + (data.piece_sides[r][c] ?? 'no piece') + ' · ' + data.terrain[r][c];
    if (data.board[r][c] === 'ambiguous_royal') $('cellInfo').textContent += ' · Select queen or white king, or exclude this cell from training.';
}
function labelView() {
    $('grid').classList.toggle('show-labels', $('showLabels').checked);
}
$('showLabels').onchange = labelView;
labelView();
function cropValue() { const parts = $('crop').value.split(',').map(x => x.trim()); if (parts.length !== 4 || parts.some(x => !/^\d+$/.test(x))) throw Error('Enter four nonnegative integer crop coordinates'); return parts.map(Number) }
function preview() { const crop = cropValue(); $('image').src = '/api/board?id=' + encodeURIComponent($('items').value) + '&crop=' + encodeURIComponent(JSON.stringify(crop)); if (data) draw(); }
function renderSoulSlots(souls) {
    $('soulSlots').replaceChildren();
    for (const piece of souls || []) {
        const row = document.createElement('div');
        row.className = 'soul-slot';
        const select = document.createElement('select');
        select.setAttribute('aria-label', 'Harvested soul piece');
        const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = 'Choose piece'; select.append(placeholder);
        for (const value of soulPieces) { const option = document.createElement('option'); option.value = value; option.textContent = value.replace('_', ' '); select.append(option); }
        select.value = piece;
        select.onchange = () => { $('soulReviewed').checked = true; dirty = true; };
        const remove = document.createElement('button');
        remove.type = 'button'; remove.textContent = '×'; remove.setAttribute('aria-label', 'Remove soul');
        remove.onclick = () => { row.remove(); $('soulReviewed').checked = true; dirty = true; };
        row.append(select, remove); $('soulSlots').append(row);
    }
}
function soulValues() { return [...$('soulSlots').querySelectorAll('select')].map(select => select.value) }
async function load() { try { const a = await api('/api/item?id=' + encodeURIComponent($('items').value)); data = a.data; if (!Number.isInteger(data.moat_row)) throw Error('Restart the review server to use moat card controls.'); revision = a.revision; $('moatActive').value = data.moat_active === true ? 'true' : data.moat_active === false ? 'false' : ''; data.exclude_cells ??= []; data.screen_state ??= {}; data.screen_state.locked_cells ??= []; data.screen_state.reviews ??= {}; $('screenKind').value = data.screen_state.screen_kind ?? ''; for (const [key, id] of Object.entries(ammoInputs)) $(id).value = data.screen_state.ammo?.[key] ?? ''; for (const key of combatFields) $(key).value = data.screen_state.combat_stats?.[key] ?? ''; for (const [section, id] of Object.entries({ ...reviewInputs, ...independentReviewInputs })) $(id).checked = data.screen_state.reviews[section] === true; renderSoulSlots(data.screen_state.souls); $('fullImage').src = '/api/screen?id=' + encodeURIComponent($('items').value); $('fullLink').href = $('fullImage').src; selected = [0, 0]; $('floor').value = data.floor ?? ''; $('crop').value = data.game_crop.join(', '); $('split').textContent = 'Session split: ' + data.split; const item = items.find(candidate => candidate.id === currentId); item.screen_kind = data.screen_state.screen_kind; syncScreenKind(); preview(); draw(); dirty = false; showReviewStatus() } catch (e) { status(e.message, true) } }
$('image').onerror = () => status('Board preview failed. Check crop coordinates.');
$('items').onchange = () => { if (dirty && !confirm('Discard unsaved edits?')) { $('items').value = currentId; return } pendingPieceApprovalId = null; currentId = $('items').value; remember(); renderOptions(); load() }; let currentId = '';
$('sessions').onchange = () => { if (dirty && !confirm('Discard unsaved edits?')) { $('sessions').value = currentSession; return } pendingPieceApprovalId = null; currentSession = $('sessions').value; const list = sessionItems(); currentId = firstPending(list).id; remember(); renderOptions(); load() };
$('locked').onchange = () => { const [r, c] = selected; data.screen_state.locked_cells = data.screen_state.locked_cells.filter(x => x[0] !== r || x[1] !== c); if ($('locked').checked) data.screen_state.locked_cells.push([r, c]); dirty = true; draw() };
for (const id of [...Object.values(ammoInputs), ...combatFields, 'screenKind', ...Object.values(reviewInputs), ...Object.values(independentReviewInputs)]) $(id).oninput = () => { dirty = true };
$('addSoul').onclick = () => { const souls = soulValues(); souls.push(''); renderSoulSlots(souls); $('soulReviewed').checked = true; dirty = true; $('soulSlots').lastElementChild.querySelector('select').focus(); };
$('screenKind').addEventListener('change', syncScreenKind);
$('label').onchange = () => { const [r, c] = selected; data.board[r][c] = $('label').value; data.piece_sides[r][c] = data.board[r][c] === 'empty' ? null : data.board[r][c] === 'player_king' ? 'ally' : data.piece_sides[r][c] ?? 'enemy'; dirty = true; draw() };
$('pieceSide').onchange = () => { const [r, c] = selected; data.piece_sides[r][c] = $('pieceSide').value || 'unknown'; dirty = true; draw() };
$('moatActive').onchange = () => { const value = $('moatActive').value; data.moat_active = value === '' ? null : value === 'true'; data.terrain = Array.from({ length: 8 }, (_, row) => Array(8).fill(data.moat_active === null ? 'unknown' : data.moat_active && row === data.moat_row ? 'moat' : 'normal')); dirty = true; draw() };
$('excluded').onchange = () => { const [r, c] = selected; data.exclude_cells = data.exclude_cells.filter(x => x[0] !== r || x[1] !== c); if ($('excluded').checked) data.exclude_cells.push([r, c]); dirty = true; draw() };
for (const id of ['floor', 'crop']) $(id).oninput = () => { dirty = true };
$('preview').onclick = () => { try { preview() } catch (e) { status(e.message) } };
async function save(confirmed) {
    const buttons = [$('draft'), $('approve')];
    buttons.forEach(button => { button.disabled = true; });
    try {
        const raw = $('floor').value;
        const floor = raw === '' ? null : Number(raw);
        const invalidFloor = floor !== null && (!Number.isInteger(floor) || floor < 1);
        if (currentReviewTab === 'pieces' && confirmed && (floor === null || invalidFloor)) {
            pendingPieceApprovalId = currentId;
            switchReviewTab('screen');
            $('floor').setCustomValidity('Enter a positive floor number before approval.');
            $('floor').reportValidity();
            $('floor').focus();
            status('Enter the floor, then save page info to approve these pieces and continue.');
            return;
        }
        if (invalidFloor) {
            $('floor').setCustomValidity('Enter a positive floor number.');
            $('floor').reportValidity();
            $('floor').focus();
            throw Error('Enter a positive floor number.');
        }
        $('floor').setCustomValidity('');
        const ammo = Object.fromEntries(Object.entries(ammoInputs).map(([key, id]) => [key, $(id).value === '' ? null : Number($(id).value)]));
        const combat_stats = Object.fromEntries(combatFields.map(key => [key, $(key).value === '' ? null : Number($(key).value)]));
        if ([...Object.values(ammo), ...Object.values(combat_stats)].some(value => value !== null && (!Number.isInteger(value) || value < 0))) throw Error('Numeric values must be nonnegative integers.');
        const reviews = { ...data.screen_state.reviews, ...Object.fromEntries(Object.entries({ ...reviewInputs, ...independentReviewInputs }).map(([section, id]) => [section, $(id).checked])) };
        const screen_kind = $('screenKind').value || null;
        if (reviews.screen_kind && screen_kind === null) throw Error('Select a screen kind before marking it reviewed.');
        const cards = data.screen_state.cards ?? { left: null, right: null };
        const selectedSouls = soulValues();
        if (reviews.soul && selectedSouls.some(piece => !piece)) throw Error('Select a piece for every harvested soul slot.');
        const souls = reviews.soul ? selectedSouls : null;
        data.screen_state = { screen_kind, ammo, combat_stats, cards, card_choices: data.screen_state.card_choices ?? null, card_reviewed_slots: data.screen_state.card_reviewed_slots ?? [], souls, locked_cells: data.screen_state.locked_cells, reviews };
        const completingPieceApproval = currentReviewTab === 'screen' && pendingPieceApprovalId === currentId;
        const boardConfirmed = currentReviewTab === 'pieces' ? confirmed : completingPieceApproval ? true : data.confirmed;
        const pieceReviewed = (currentReviewTab === 'pieces' && confirmed) || completingPieceApproval;
        if (boardConfirmed && screen_kind === 'gameplay') {
            for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
                if (data.board[r][c] === 'ambiguous_royal' && !exclude(r, c)) {
                    selected = [r, c];
                    switchReviewTab('pieces');
                    draw();
                    $('label').focus();
                    throw Error('Select queen or white king for the highlighted cell, or exclude it from training.');
                }
            }
        }
        status('Saving review…');
        const result = await api('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Review-Token': token },
            body: JSON.stringify({ id: currentId, revision, screen_state: data.screen_state, board: data.board, piece_sides: data.piece_sides, terrain: data.terrain, moat_active: data.moat_active, exclude_cells: data.exclude_cells, game_crop: cropValue(), floor, split: data.split, confirmed: boardConfirmed, piece_reviewed: pieceReviewed })
        });
        revision = result.revision;
        data.confirmed = result.confirmed;
        data.piece_review = result.piece_review;
        dirty = false;
        const item = items.find(candidate => candidate.id === currentId);
        item.confirmed = result.confirmed;
        item.piece_review = result.piece_review;
        item.screen_kind = screen_kind;
        item.reviews = reviews;
        renderOptions();
        if (completingPieceApproval) {
            pendingPieceApprovalId = null;
            if (!result.confirmed) {
                status('Page information saved; this screen does not require piece review.');
                return;
            }
            switchReviewTab('pieces');
            const nextPiece = nextPending();
            if (nextPiece) await moveTo(nextPiece);
            else status('Piece review complete — all applicable screens approved.');
            return;
        }
        if (currentReviewTab === 'pieces') pendingPieceApprovalId = null;
        if (!confirmed) {
            const message = currentReviewTab === 'pieces'
                ? 'Piece draft saved; excluded from approved datasets.'
                : nonGameplay(screen_kind)
                    ? 'Page information saved; this screen is excluded from piece review.'
                    : 'Page information saved; piece approval is unchanged.';
            status(message);
            return;
        }
        const next = nextPending();
        if (next) {
            await moveTo(next);
        } else {
            const remaining = items.filter(needsReview).length;
            if (remaining) status('No later unreviewed screen — ' + remaining + ' earlier screen' + (remaining === 1 ? '' : 's') + ' still need review.');
            else status(currentReviewTab === 'pieces' ? 'Piece review complete — all ' + items.length + ' screens approved.' : 'Page-information review complete — all ' + items.length + ' screens reviewed.');
        }
    } catch (error) {
        status('Not saved — ' + error.message, true);
    } finally {
        buttons.forEach(button => { button.disabled = false; });
    }
}
$('floor').addEventListener('input', () => $('floor').setCustomValidity(''));
$('toggleFields').onclick = () => {
    const hidden = $('gameplayFields').classList.toggle('fields-hidden');
    $('toggleFields').textContent = hidden ? 'Show fields' : 'Hide fields';
    $('toggleFields').setAttribute('aria-pressed', String(hidden));
};
$('draft').onclick = () => save(false); $('approve').onclick = () => save(true);
function renderOptions() { const list = sessionItems(); $('items').replaceChildren(); for (const item of list) { const o = document.createElement('option'); o.value = item.id; o.textContent = reviewMarker(item) + ' ' + item.id.split('/').pop(); $('items').append(o) } $('items').value = currentId; const index = list.findIndex(x => x.id === currentId); $('previous').disabled = index <= 0; $('next').disabled = index >= list.length - 1 && !otherSessionPending(); const count = section => list.filter(item => item.reviews?.[section]).length; const pieceItems = list.filter(pieceApplicable); $('progress').textContent = currentReviewTab === 'pieces' ? 'pieces ' + pieceItems.filter(x => x.confirmed).length + '/' + pieceItems.length + ' · screenshot ' + (index + 1) + '/' + list.length : 'page info ' + list.filter(pageComplete).length + '/' + list.length + ' · kind ' + count('screen_kind') + ' · ammo ' + count('ammo') + ' · stats ' + count('combat_stats') + ' · souls ' + count('soul') + ' · screenshot ' + (index + 1) + '/' + list.length }
async function navigate(delta) { const list = sessionItems(); const index = list.findIndex(x => x.id === currentId); const target = delta < 0 ? list[index - 1] : list[index + 1] || otherSessionPending(); if (!target) return; if (dirty && !confirm('Discard unsaved edits?')) return; await moveTo(target) }
$('previous').onclick = () => navigate(-1); $('next').onclick = () => navigate(1);
window.onbeforeunload = e => { if (dirty) { e.preventDefault(); e.returnValue = '' } };
(async () => { try { items = await api('/api/items'); if (!items.length) { status('No annotations. Run with --session to prepare inbox images.'); return } const sessions = [...new Set(items.map(x => x.session))]; const saved = recalled(); const requestedSession = new URLSearchParams(location.search).get('session'); currentSession = sessions.includes(requestedSession) ? requestedSession : sessions.includes(saved.session) ? saved.session : sessions[0]; for (const session of sessions) { const o = document.createElement('option'); o.value = session; o.textContent = session; $('sessions').append(o) } $('sessions').value = currentSession; const list = sessionItems(); currentId = list.some(x => x.id === saved.id) ? saved.id : firstPending(list).id; if (requestedSession) history.replaceState(null, '', location.pathname + location.hash); remember(); renderOptions(); await load() } catch (e) { status(e.message) } })();

$('predictState').onclick = async () => {
    $('predictState').disabled = true;
    try {
        const result = await api('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Review-Token': token }, body: JSON.stringify({ id: currentId, revision, game_crop: cropValue() }) });
        let count = 0;
        if ($('screenKind').value === '' && result.screen_state.screen_kind !== null) { $('screenKind').value = result.screen_state.screen_kind; $('screenKindReviewed').checked = false; syncScreenKind(); count++; }
        if ($('floor').value === '' && result.floor !== null) { $('floor').value = result.floor; count++; }
        for (const [key, id] of Object.entries(ammoInputs)) if ($(id).value === '' && result.screen_state.ammo[key] !== null) { $(id).value = result.screen_state.ammo[key]; count++; $('ammoReviewed').checked = false; }
        for (const key of combatFields) if ($(key).value === '' && result.screen_state.combat_stats[key] !== null) { $(key).value = result.screen_state.combat_stats[key]; count++; $('combatReviewed').checked = false; }
        if (!$('soulReviewed').checked && result.screen_state.souls !== null) { renderSoulSlots(result.screen_state.souls); $('soulReviewed').checked = false; count++; }
        if (count) dirty = true;
        status(count ? 'Filled ' + count + ' fields. Review predictions before saving.' : 'No confident matches. Add reviewed examples with known values first.');
    } catch (error) { status('Prediction failed — ' + error.message, true); }
    finally { $('predictState').disabled = false; }
};

function switchReviewTab(name, focus = false) {
    if (name === 'pieces' && $('piecesTab').disabled) name = 'screen';
    currentReviewTab = name;
    const main = document.querySelector('main');
    main.classList.toggle('page-info-mode', name === 'screen');
    for (const [tab, panel, key] of [['piecesTab', 'piecesView', 'pieces'], ['screenTab', 'screenView', 'screen']]) {
        const active = name === key;
        $(tab).setAttribute('aria-selected', String(active));
        $(tab).tabIndex = active ? 0 : -1;
        $(panel).hidden = !active;
        $(key === 'pieces' ? 'piecesControls' : 'screenControls').hidden = !active;
        if (active && focus) $(tab).focus();
    }
    $('draft').textContent = name === 'pieces' ? 'Save draft' : 'Save page info';
    $('approve').textContent = name === 'pieces' ? 'Approve pieces ✓' : 'Save & next →';
    $('saveHelp').textContent = name === 'pieces' ? 'Check every piece label before approval.' : 'Save page information and harvested souls, then continue to the next unfinished screen.';
    if (items.length) { renderOptions(); showReviewStatus(); }
}
function syncScreenKind() {
    const unavailable = nonGameplay($('screenKind').value || null);
    $('gameplayFields').hidden = unavailable;
    $('piecesTab').disabled = unavailable;
    if (unavailable && currentReviewTab === 'pieces') switchReviewTab('screen');
    if (items.length) {
        const item = items.find(candidate => candidate.id === currentId);
        if (item) item.screen_kind = $('screenKind').value || null;
        renderOptions();
    }
}
for (const [id, name] of [['piecesTab', 'pieces'], ['screenTab', 'screen']]) {
    $(id).onclick = () => {
        switchReviewTab(name);
        history.replaceState(null, '', name === 'screen' ? '#page-info' : '#pieces');
    };
    $(id).onkeydown = event => {
        if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
            event.preventDefault();
            switchReviewTab(event.key === 'Home' ? 'pieces' : event.key === 'End' ? 'screen' : name === 'pieces' ? 'screen' : 'pieces', true);
        }
    };
}
switchReviewTab(location.hash === '#page-info' ? 'screen' : 'pieces');
