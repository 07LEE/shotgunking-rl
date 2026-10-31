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
        check(!document.querySelector('select#sessions, select#items'), 'selection dropdowns stay removed');
        check(!row(id('play', 'a-complete')), 'completed screenshots stay hidden');
        const originalId = currentId;
        queueFor('missing-floor').querySelector('summary').click();
        check(currentId === originalId, 'expanding a session does not navigate');
        await task('combat_stats');
        check(shown('attack') && !shown('ammo_loaded') && !shown('screenKind') && !shown('addSoul') && !shown('label'), 'only the selected task fields are visible');
        check(!document.querySelector('.screen-overview input'), 'screenshot has no overlaid inputs');
        check($('reviewSectionStatus').children.length === 1, 'only the active review status is shown');
        check($('ammo_loaded').disabled && !$('attack').disabled && $('screenKind').disabled, 'saved section locks are preserved');
        const originalApi = api;
        api = async (url, ...args) => url === '/api/predict' ? { floor: null, screen_state: { screen_kind: null, ammo: { loaded: 9, reserve: 9 }, combat_stats: { ...Object.fromEntries(combatFields.map(key => [key, null])), range_min: 3 }, souls: null } } : originalApi(url, ...args);
        await $('predictState').onclick(); api = originalApi;
        check($('ammo_loaded').value === '2' && $('range_min').value === '3' && $('attack').value === '4', 'auto-fill preserves reviewed and already entered values');
        window.confirm = () => false;
        row(id('play', 'c-pending')).click();
        check(currentId === originalId && $('range_min').value === '3', 'canceling image navigation preserves unsaved work');
        await $('ammoTab').onclick();
        check(currentReviewTab === 'combat_stats' && $('range_min').value === '3', 'canceling task navigation preserves unsaved work');
        window.confirm = () => true;
        $('showCompleted').checked = true; await $('showCompleted').onchange(); await idle();
        await pick(id('play', 'a-complete'));
        check($('attack').disabled && $('approve').disabled && $('draft').disabled, 'completed task is read-only');
        const oldRevision = revision; await save(false);
        check(revision === oldRevision, 'read-only review cannot be saved accidentally');
        await task('pieces');
        check($('label').disabled && Array.from($('grid').children).every(cell => cell.disabled), 'approved board remains locked');
        await task('ammo');
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
        await $('approve').onclick(); await idle();
        saved = await stored(id('finish-page', 'finish-page'));
        check(saved.screen_state.reviews.screen_kind && !saved.screen_state.reviews.ammo && !saved.screen_state.reviews.combat_stats && !saved.screen_state.reviews.soul && !saved.confirmed, 'completion approves only the visible task');
        check(!row(id('finish-page', 'finish-page')), 'completed task leaves its pending list');
        items = await api('/api/items'); renderOptions();
        check(!row(id('finish-page', 'finish-page')), 'completion stays removed after reading saved state');
        await task('ammo');
        check(Boolean(row(id('finish-page', 'finish-page'))), 'unfinished sections of the same screenshot remain available');
        await pick(id('finish-page', 'finish-page'));
        check(shown('ammo_loaded') && shown('ammo_reserve') && !shown('attack') && !shown('floor'), 'ammunition view shows only its two fields');
        await save(false);
        check(Boolean(row(id('finish-page', 'finish-page'))), 'saving progress does not complete a task');
        await $('approve').onclick(); await idle();
        saved = await stored(id('finish-page', 'finish-page'));
        check(saved.screen_state.reviews.ammo && !saved.screen_state.reviews.combat_stats && !saved.screen_state.reviews.soul && !row(id('finish-page', 'finish-page')), 'ammunition completion leaves hidden sections unapproved');
        await task('screen_kind'); await pick(id('finish-card', 'finish-card')); await $('approve').onclick(); await idle();
        saved = await stored(id('finish-card', 'finish-card'));
        check(saved.screen_state.reviews.screen_kind && !saved.screen_state.reviews.ammo && !row(id('finish-card', 'finish-card')), 'non-gameplay screen classification completes normally');
        await task('ammo');
        check(!row(id('finish-card', 'finish-card')), 'non-gameplay screenshots do not enter ammunition review');
        await task('screen_kind'); await pick(id('finish-unknown', 'finish-unknown'));
        const beforeUnknown = revision; await $('approve').onclick();
        check(revision === beforeUnknown && row(id('finish-unknown', 'finish-unknown')) && $('status').textContent.includes('Not saved'), 'invalid completion remains pending with an error');
        await task('soul'); await pick(id('finish-page', 'finish-page'));
        $('addSoul').click(); const beforeSoul = revision; await $('approve').onclick();
        check(revision === beforeSoul && row(id('finish-page', 'finish-page')), 'unidentified soul slot blocks completion');
        $('soulSlots').querySelector('button').click(); await $('approve').onclick(); await idle();
        saved = await stored(id('finish-page', 'finish-page'));
        check(saved.screen_state.reviews.soul && saved.screen_state.souls.length === 0 && !saved.screen_state.reviews.combat_stats, 'empty soul list can be reviewed without approving combat stats');
        document.body.innerHTML = '<pre id="browser-test-result">' + JSON.stringify({ ok: true, checks }) + '</pre>';
    } catch (error) { document.body.innerHTML = '<pre id="browser-test-result">' + JSON.stringify({ ok: false, checks, error: error.stack }) + '</pre>'; }
})();
