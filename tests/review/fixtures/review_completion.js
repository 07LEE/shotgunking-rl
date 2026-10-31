(async () => {
    const checks = [];
    function check(ok, name) { if (!ok) throw Error(name); checks.push(name); }
    async function idle() {
        for (let i = 0; i < 300; i++) {
            if (data && !loading && loadedId === currentId) return;
            await new Promise(resolve => setTimeout(resolve, 20));
        }
        throw Error('Load timeout');
    }
    const queueFor = session => Array.from(document.querySelectorAll('.queue-session')).find(group => group.dataset.session === session);
    const row = id => Array.from(document.querySelectorAll('.queue-item')).find(button => button.dataset.id === id);
    const id = (session, name) => 'sessions/' + session + '/annotations/' + name + '.png.json';
    async function pick(target) {
        const button = row(target);
        if (!button) throw Error('Missing queue item ' + target);
        const group = button.closest('details');
        if (!group.open) group.querySelector('summary').click();
        button.click(); await idle();
    }
    async function task(key) { await chooseTask(key); await idle(); }
    async function stored(target) { return (await api('/api/item?id=' + encodeURIComponent(target))).data; }
    const shown = input => $(input).getClientRects().length > 0;
    window.confirm = () => true;
    try {
        await idle();
        check(currentId === id('play', 'b-partial'), 'starts at unfinished screenshot');
        check(document.querySelector('.review-sidebar').getBoundingClientRect().right < document.querySelector('.review-content').getBoundingClientRect().left, 'review queue stays on the left');
        check(!document.querySelector('select#sessions, select#items') && !document.querySelector('.review-sidebar .review-tabs') && document.querySelector('.review-type .review-tabs'), 'review types stay separate from the image queue without selection dropdowns');
        check(!row(id('play', 'a-complete')), 'completed screenshots stay hidden');
        const originalId = currentId;
        queueFor('missing-floor').querySelector('summary').click();
        check(currentId === originalId, 'expanding a session does not navigate');
        await task('screen_kind');
        check(shown('attack') && shown('ammo_loaded') && shown('screenKind') && shown('addSoul') && !shown('label'), 'all screen information fields share one view');
        check(!document.querySelector('.screen-overview input'), 'screenshot has no overlaid inputs');
        check($('reviewSectionStatus').children.length === 4, 'section approvals remain individually visible');
        check($('ammo_loaded').disabled && !$('attack').disabled && $('screenKind').disabled, 'saved section locks are preserved');
        const originalApi = api;
        api = async (url, ...args) => url === '/api/predict' ? { floor: null, screen_state: { screen_kind: null, ammo: { loaded: 9, reserve: 9 }, combat_stats: { ...Object.fromEntries(combatFields.map(key => [key, null])), range_min: 3 }, souls: null } } : originalApi(url, ...args);
        await $('predictState').onclick(); api = originalApi;
        check($('ammo_loaded').value === '2' && $('range_min').value === '3' && $('attack').value === '4', 'auto-fill preserves reviewed and already entered values');
        window.confirm = () => false;
        row(id('play', 'c-pending')).click();
        check(currentId === originalId && $('range_min').value === '3', 'canceling image navigation preserves unsaved work');
        await $('piecesTab').onclick();
        check(currentReviewTab === 'screen_kind' && $('range_min').value === '3', 'canceling task navigation preserves unsaved work');
        window.confirm = () => true;
        $('showCompleted').checked = true; await $('showCompleted').onchange(); await idle();
        await pick(id('play', 'a-complete'));
        check($('attack').disabled && $('approve').disabled && $('draft').disabled, 'completed task is read-only');
        const oldRevision = revision; await save(false);
        check(revision === oldRevision, 'read-only review cannot be saved accidentally');
        await task('pieces');
        check($('label').disabled && Array.from($('grid').children).every(cell => cell.disabled), 'approved board remains locked');
        await task('screen_kind');
        document.querySelector('[aria-label="Edit reviewed ammunition"]').click();
        check(!$('ammo_loaded').disabled && !$('ammoReviewed').checked && dirty, 'explicit edit reopens the selected task');
        check($('attack').disabled && $('screenKind').disabled, 'other reviewed sections stay locked');
        $('ammo_loaded').value = '1'; $('ammo_loaded').oninput(); await save(false);
        let saved = await stored(id('play', 'a-complete'));
        check(saved.confirmed && saved.screen_state.reviews.screen_kind && !saved.screen_state.reviews.ammo && saved.screen_state.ammo.loaded === 1, 'progress save preserves unrelated approvals');
        await $('approve').onclick(); await idle();
        saved = await stored(id('play', 'a-complete'));
        check(saved.screen_state.reviews.ammo, 'complete action restores the edited approval');
        $('showCompleted').checked = false; await $('showCompleted').onchange(); await idle();
        check(!row(id('play', 'a-complete')) && !queueFor('finished'), 'completed work is absent from the pending queue');
        check(document.querySelector('.queue-item[aria-current="true"]').dataset.id === currentId, 'sidebar marks the selected image');
        await task('pieces'); await pick(id('play', 'c-pending')); await save(true); await idle();
        check(currentId === id('play', 'b-partial'), 'next pending wraps to earlier unfinished pieces');
        await pick(id('missing-floor', 'no-floor')); await save(true);
        check(currentReviewTab === 'screen_kind' && shown('floor') && !$('floor').disabled && !$('approve').disabled, 'piece approval exposes missing floor input');
        $('floor').value = '2'; $('floor').oninput(); await save(true); await idle();
        saved = await stored(id('missing-floor', 'no-floor'));
        check(saved.floor === 2 && saved.confirmed, 'floor hand-off completes piece approval');
        await task('screen_kind'); await pick(id('finish-page', 'finish-page'));
        check(document.querySelectorAll('.review-tabs [role="tab"]').length === 2, 'only pieces and screen info are selectable');
        await save(false);
        check(Boolean(row(id('finish-page', 'finish-page'))), 'draft keeps the image pending');
        $('addSoul').click(); const beforeSoul = revision; await save(true);
        check(revision === beforeSoul && row(id('finish-page', 'finish-page')), 'unknown soul blocks grouped completion');
        $('soulSlots').querySelector('button').click(); await save(true); await idle();
        saved = await stored(id('finish-page', 'finish-page'));
        check(['screen_kind', 'ammo', 'combat_stats', 'soul'].every(key => saved.screen_state.reviews[key]) && !saved.confirmed, 'screen info completes all applicable sections without approving pieces');
        check(!row(id('finish-page', 'finish-page')), 'completed screen info leaves queue');
        items = await api('/api/items'); renderOptions();
        check(!row(id('finish-page', 'finish-page')), 'completion remains hidden after reload');
        await pick(id('finish-card', 'finish-card'));
        check(shown('screenKind') && !shown('attack') && !shown('addSoul'), 'non-gameplay only shows classification');
        await save(true); await idle(); saved = await stored(id('finish-card', 'finish-card'));
        check(saved.screen_state.reviews.screen_kind && !saved.screen_state.reviews.ammo, 'non-gameplay completion preserves gameplay reviews');
        await pick(id('finish-unknown', 'finish-unknown')); const beforeUnknown = revision; await save(true);
        check(revision === beforeUnknown && row(id('finish-unknown', 'finish-unknown')), 'unknown kind blocks completion');
        document.body.innerHTML = '<pre id="browser-test-result">' + JSON.stringify({ ok: true, checks }) + '</pre>';
    } catch (error) { document.body.innerHTML = '<pre id="browser-test-result">' + JSON.stringify({ ok: false, checks, error: error.stack }) + '</pre>'; }
})();
