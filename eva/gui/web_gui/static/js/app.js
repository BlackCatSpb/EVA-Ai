/* CogniFlex Web GUI — app.js */
(function () {
    'use strict';

    /* ── State ── */
    let token = null;
    let userId = null;
    let sessions = [];
    let activeSessionId = null;
    let sidebarOpen = false;

    const $ = (s, p) => (p || document).querySelector(s);
    const $$ = (s, p) => [...(p || document).querySelectorAll(s)];

    /* ── API ── */
    async function api(path, opts = {}) {
        const headers = { 'Content-Type': 'application/json' };
        if (userId) headers['X-User-ID'] = userId;
        const r = await fetch(`/api${path}`, {
            ...opts,
            headers: { ...headers, ...opts.headers },
            body: opts.body ? JSON.stringify(opts.body) : undefined
        });
        return r.json();
    }

    /* ── Toast ── */
    function toast(msg, type = 'info') {
        const c = $('#toastContainer');
        const t = document.createElement('div');
        t.className = `toast ${type}`;
        t.textContent = msg;
        c.appendChild(t);
        setTimeout(() => t.remove(), 3000);
    }

    /* ── Login ── */
    $('#loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const u = $('#loginUsername').value.trim();
        const p = $('#loginPassword').value.trim();
        if (!u || !p) return;

        try {
            const d = await api('/login', { method: 'POST', body: { username: u, password: p } });
            if (d.error) {
                $('#loginError').textContent = d.error;
                return;
            }
            token = true;
            userId = d.sessions[0]?.user_id;
            sessions = d.sessions || [];
            activeSessionId = d.session_id;
            $('#loginScreen').style.display = 'none';
            $('#appContainer').style.display = 'flex';
            $('#userName').textContent = d.user;
            $('#settingsUser').textContent = d.user;
            renderSessions();
            showWelcome();
            toast('Добро пожаловать', 'success');
        } catch {
            $('#loginError').textContent = 'Ошибка подключения';
        }
    });

    /* ── Logout ── */
    $('#logoutBtn').addEventListener('click', () => {
        token = null;
        userId = null;
        sessions = [];
        activeSessionId = null;
        $('#appContainer').style.display = 'none';
        $('#loginScreen').style.display = 'flex';
        $('#loginUsername').value = '';
        $('#loginPassword').value = '';
        $('#chatMessages').innerHTML = '';
    });

    /* ── Sessions ── */
    function renderSessions() {
        const c = $('#sessionList');
        c.innerHTML = '';
        sessions.forEach(s => {
            const el = document.createElement('div');
            el.className = `session-item${s.id === activeSessionId ? ' active' : ''}`;
            el.innerHTML = `
                <span class="session-name">${esc(s.name || 'Без названия')}</span>
                <button class="session-del" data-id="${s.id}" title="Удалить">×</button>
            `;
            el.addEventListener('click', (e) => {
                if (e.target.classList.contains('session-del')) return;
                selectSession(s.id);
            });
            el.querySelector('.session-del').addEventListener('click', () => deleteSession(s.id));
            c.appendChild(el);
        });
    }

    async function selectSession(id) {
        activeSessionId = id;
        renderSessions();
        await loadSessionMessages(id);
    }

    async function deleteSession(id) {
        sessions = sessions.filter(s => s.id !== id);
        if (activeSessionId === id) {
            activeSessionId = sessions[0]?.id || null;
        }
        await api('/sessions', { method: 'DELETE', body: { session_id: id } });
        renderSessions();
        if (activeSessionId) {
            await loadSessionMessages(activeSessionId);
        } else {
            $('#chatMessages').innerHTML = '';
            showWelcome();
        }
    }

    $('#newChatBtn').addEventListener('click', async () => {
        const d = await api('/sessions', { method: 'POST', body: { name: 'Новый чат' } });
        sessions = d.sessions || [];
        activeSessionId = d.session_id;
        renderSessions();
        $('#chatMessages').innerHTML = '';
        showWelcome();
    });

    /* ── Messages ── */
    function showWelcome() {
        const c = $('#chatMessages');
        if (c.children.length === 0) {
            c.innerHTML = `<div class="welcome"><h2>ЕВА</h2><p>Напишите сообщение, чтобы начать</p></div>`;
        }
    }

    async function loadSessionMessages(id) {
        try {
            const d = await api(`/session/${id}`);
            const c = $('#chatMessages');
            c.innerHTML = '';
            if (!d.context || d.context.length === 0) {
                showWelcome();
                return;
            }
            d.context.forEach(node => {
                if (node.user_message) addMsg('user', node.user_message, node.entities);
                if (node.assistant_message) addMsg('system', node.assistant_message);
            });
        } catch { /* ignore */ }
    }

    function addMsg(role, text, entities, reasoning, fileData) {
        const c = $('#chatMessages');
        const welcome = c.querySelector('.welcome');
        if (welcome) welcome.remove();

        const div = document.createElement('div');
        div.className = 'msg';

        let reasoningHtml = '';
        let fileHtml = '';
        
        // Show file if attached
        if (fileData && fileData.filename) {
            fileHtml = `
                <div class="msg-file">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
                    <span>${esc(fileData.filename)}</span>
                </div>
            `;
        }
        
        // Use reasoning from brain if available
        if (reasoning) {
            const reasoningText = typeof reasoning === 'string' ? reasoning : JSON.stringify(reasoning, null, 2);
            reasoningHtml = `
                <div class="msg-reasoning">
                    <button class="reasoning-toggle" onclick="this.classList.toggle('open')">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                        Рассуждения
                    </button>
                    <div class="reasoning-body"><pre>${esc(reasoningText)}</pre></div>
                </div>
            `;
        } else if (entities && entities.length > 0) {
            // Fallback to entity-based reasoning
            const steps = entities.map((e, i) =>
                `${i + 1}. [${e.type}] ${e.keyword}`
            ).join('\n');
            reasoningHtml = `
                <div class="msg-reasoning">
                    <button class="reasoning-toggle" onclick="this.classList.toggle('open')">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                        Рассуждения
                    </button>
                    <div class="reasoning-body"><pre>${esc(steps)}</pre></div>
                </div>
            `;
        }

        const roleLabel = role === 'user' ? 'Вы' : 'ЕВА';
        const roleClass = role;

        div.innerHTML = `
            <div class="msg-inner">
                <div class="msg-role ${roleClass}">${roleLabel}</div>
                <div class="msg-text">${formatText(text)}</div>
                ${fileHtml}
                ${reasoningHtml}
            </div>
        `;
        c.appendChild(div);
        c.scrollTop = c.scrollHeight;
    }

    function addTyping() {
        const c = $('#chatMessages');
        const div = document.createElement('div');
        div.className = 'msg';
        div.id = 'typingIndicator';
        div.innerHTML = `
            <div class="msg-inner">
                <div class="msg-role system">ЕВА</div>
                <div class="msg-text typing-dots"><span>·</span><span>·</span><span>·</span></div>
            </div>
        `;
        c.appendChild(div);
        c.scrollTop = c.scrollHeight;
    }

    function removeTyping() {
        const el = $('#typingIndicator');
        if (el) el.remove();
    }

    /* ── File Upload ── */
    let currentFile = null;
    let currentFileData = null;

    $('#attachBtn').addEventListener('click', () => {
        $('#fileInput').click();
    });

    $('#fileInput').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        currentFile = file;
        $('#fileName').textContent = file.name;
        $('#filePreview').style.display = 'flex';

        // Upload file to server
        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();
            if (data.error) {
                toast(data.error, 'error');
                clearFile();
                return;
            }
            currentFileData = data;
            toast(`Файл "${file.name}" загружен`, 'success');
        } catch (err) {
            toast('Ошибка загрузки файла', 'error');
            clearFile();
        }
    });

    $('#removeFile').addEventListener('click', clearFile);

    function clearFile() {
        currentFile = null;
        currentFileData = null;
        $('#fileInput').value = '';
        $('#filePreview').style.display = 'none';
    }

    /* ── Memory ── */
    function loadMemory() {
        api('/memory-graph').then(data => {
            if (data.error) {
                toast('Ошибка загрузки памяти', 'error');
                return;
            }
            const stats = data.stats || {};
            $('#totalNodes').textContent = stats.total_nodes || data.nodes?.length || 0;
            $('#totalEdges').textContent = stats.total_edges || data.edges?.length || 0;
            
            // Simple list view of nodes
            const graphEl = $('#memoryGraph');
            if (data.nodes && data.nodes.length > 0) {
                graphEl.innerHTML = data.nodes.slice(0, 50).map(n => 
                    `<div class="node-item">${esc(n.label || n.id)}</div>`
                ).join('');
            } else {
                graphEl.innerHTML = '<div class="empty-state">Нет данных о памяти</div>';
            }
        }).catch(() => {
            toast('Ошибка загрузки памяти', 'error');
        });
    }
    
    $('#refreshMemory')?.addEventListener('click', loadMemory);

    /* ── Chat ── */
    function sendMessage() {
        const input = $('#chatInput');
        const text = input.value.trim();
        if ((!text && !currentFileData) || !activeSessionId) return;

        // Show file in message if attached
        let msgText = text;
        if (currentFileData) {
            msgText = text || `Проанализируй файл ${currentFileData.filename}`;
        }

        input.value = '';
        input.style.height = 'auto';
        addMsg('user', msgText, null, null, currentFileData);
        
        // Рассуждения - только в folded меню
        addTyping();
        const body = { message: text || `Проанализируй файл ${currentFileData?.filename || ''}`, session_id: activeSessionId, user_id: userId };
        if (currentFileData) {
            body.file_data = currentFileData;
        }

        api('/chat', {
            method: 'POST',
            body: body
        }).then(d => {
            removeTyping();
            
            // Рассуждения в folded меню
            // Объединяем все рассуждения в одно меню
            let allReasoning = '';
            
            // 1. Основные рассуждения (steps)
            if (d.reasoning_steps && d.reasoning_steps.length > 0) {
                allReasoning += '🔬 Этапы рассуждений:\n';
                d.reasoning_steps.forEach((step, idx) => {
                    const icons = {'generation': '💭', 'contradiction_check': '⚖️', 'ethics_check': '⚡', 'web_search': '🌐', 'refinement': '🔄', 'final_synthesis': '✨', 'document_analysis': '📄'};
                    const icon = icons[step.phase] || '•';
                    allReasoning += step.step + '. ' + icon + ' [' + step.phase + '] ' + step.thought + ' (conf: ' + step.confidence.toFixed(2) + ')\n';
                });
            }
            
            // 2. Веб-поиск
            if (d.web_search_info) {
                allReasoning += '\n🌐 Результаты поиска:\n' + d.web_search_info + '\n';
            }
            
            // 3. Самодиалог
            if (d.self_dialog) {
                const sd = d.self_dialog;
                allReasoning += '\n🔄 Самодиалог:\n';
                allReasoning += 'Тема: ' + (sd.topic || '') + '\n';
                allReasoning += 'Исход: ' + (sd.outcome || '') + '\n';
                if (sd.gaps && sd.gaps.length > 0) allReasoning += 'Пробелы: ' + sd.gaps.join(', ') + '\n';
                if (sd.actions && sd.actions.length > 0) allReasoning += 'Действия: ' + sd.actions.join(', ') + '\n';
            }
            
            // Добавляем финальный ответ с объединенными рассуждениями
            addMsg('system', d.response || 'Нет ответа', null, allReasoning || d.reasoning);
            
            // Показываем уточняющий вопрос если есть
            if (d.clarification_question) {
                const clarHtml = `
                    <div class="clarification-box">
                        <div class="clarification-icon">❓</div>
                        <div class="clarification-text">${esc(d.clarification_question)}</div>
                    </div>
                `;
                const lastMsg = c.lastElementChild;
                const lastMsgInner = lastMsg?.querySelector?.('.msg-inner');
                if (lastMsgInner) {
                    lastMsgInner.insertAdjacentHTML('beforeend', clarHtml);
                }
            }
            
            // Самодиалог теперь в меню рассуждений
            clearFile();
        }).catch(() => {
            removeTyping();
            addMsg('system', 'Ошибка соединения');
            clearFile();
        });
    }

    $('#sendBtn').addEventListener('click', sendMessage);
    $('#chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    $('#chatInput').addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    /* ── Sidebar toggle ── */
    function createSidebarToggle() {
        const btn = document.createElement('button');
        btn.className = 'sidebar-toggle';
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>`;
        btn.addEventListener('click', toggleSidebar);
        $('.main').prepend(btn);
    }

    function toggleSidebar() {
        sidebarOpen = !sidebarOpen;
        $('#sidebar').classList.toggle('collapsed', !sidebarOpen);
    }

    // Start with sidebar collapsed
    $('#sidebar').classList.add('collapsed');
    createSidebarToggle();

    // Click on sidebar area to also allow collapsing
    // Add a close button to sidebar
    const sidebarCloseBtn = document.createElement('button');
    sidebarCloseBtn.className = 'new-chat-btn';
    sidebarCloseBtn.style.marginBottom = '0';
    sidebarCloseBtn.style.border = 'none';
    sidebarCloseBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg><span>Свернуть</span>`;
    sidebarCloseBtn.addEventListener('click', toggleSidebar);
    $('#sidebar').insertBefore(sidebarCloseBtn, $('#sidebar').firstChild);

    /* ── Navigation ── */
    $$('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.nav-item').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const view = btn.dataset.view;
            $$('.view').forEach(v => v.style.display = 'none');
            $(`#${view}View`).style.display = 'flex';
            
            // Load data for specific views
            if (view === 'memory') {
                loadMemory();
            } else if (view === 'analytics') {
                loadAnalytics();
            } else if (view === 'learning') {
                loadLearning();
            }
        });
    });

    /* ── Helpers ── */
    function esc(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    function formatText(text) {
        if (!text) return '';
        let html = text;
        
        // Escape HTML first (but preserve markdown)
        html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        
        // Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
        
        // Bold and italic
        html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        html = html.replace(/___(.+?)___/g, '<strong><em>$1</em></strong>');
        html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
        html = html.replace(/_(.+?)_/g, '<em>$1</em>');
        
        // Strikethrough
        html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');
        
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Code blocks with canvas support
        html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, function(match, lang, code) {
            const langLabel = lang || 'text';
            const codeId = 'code_' + Math.random().toString(36).substr(2, 9);
            return `
                <div class="code-block" id="${codeId}">
                    <div class="code-header">
                        <span class="code-lang">${langLabel}</span>
                        <button class="code-copy" onclick="navigator.clipboard.writeText(document.getElementById('${codeId}').querySelector('code').textContent); this.textContent='✓ Скопировано'; setTimeout(() => this.textContent='Копировать', 2000);">
                            Копировать
                        </button>
                    </div>
                    <pre><code class="language-${langLabel}">${esc(code.trim())}</code></pre>
                </div>
            `;
        });
        
        // Unordered lists
        html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
        
        // Ordered lists
        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        
        // Blockquotes
        html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
        
        // Horizontal rule
        html = html.replace(/^---$/gm, '<hr>');
        
        // Line breaks
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        
        // Wrap in paragraphs
        html = '<p>' + html + '</p>';
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p>(<h[123]>)/g, '$1');
        html = html.replace(/(<\/h[123]>)<\/p>/g, '$1');
        html = html.replace(/<p>(<pre>)/g, '$1');
        html = html.replace(/(<\/pre>)<\/p>/g, '$1');
        html = html.replace(/<p>(<ul>)/g, '$1');
        html = html.replace(/(<\/ul>)<\/p>/g, '$1');
        html = html.replace(/<p>(<blockquote>)/g, '$1');
        html = html.replace(/(<\/blockquote>)<\/p>/g, '$1');
        html = html.replace(/<p>(<hr>)<\/p>/g, '$1');
        
        return html;
    }

    /* ── Status Indicator ── */
    const statusEl = $('#systemStatus');
    const statusDot = statusEl?.querySelector('.status-dot');
    const statusText = statusEl?.querySelector('.status-text');

    function updateStatus(status) {
        if (!statusEl) return;
        statusEl.className = 'status-indicator ' + status.state;
        if (statusText) statusText.textContent = status.label;
    }

    async function checkStatus() {
        try {
            const d = await api('/status');
            if (d.status === 'not_initialized') {
                updateStatus({ state: 'offline', label: 'Не подключено' });
            } else if (d.brain_connected && d.brain_running) {
                updateStatus({ state: 'online', label: 'Онлайн' });
            } else if (d.brain_connected && !d.brain_running) {
                updateStatus({ state: 'starting', label: 'Запуск...' });
            } else {
                updateStatus({ state: 'offline', label: 'Офлайн' });
            }
        } catch {
            updateStatus({ state: 'error', label: 'Ошибка' });
        }
    }

    // Poll status every 5 seconds
    checkStatus();
    setInterval(checkStatus, 5000);

    /* ── Selection Popup Menu (QWEN-style) ── */
    const selectionPopup = document.createElement('div');
    selectionPopup.className = 'selection-popup';
    selectionPopup.innerHTML = `
        <button class="popup-btn" data-action="copy" title="Копировать">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            <span>Копировать</span>
        </button>
        <div class="popup-divider"></div>
        <button class="popup-btn" data-action="ask" title="Спросить о выделенном">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            <span>Спросить</span>
        </button>
        <button class="popup-btn" data-action="explain" title="Объяснить">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
            <span>Объяснить</span>
        </button>
        <button class="popup-btn" data-action="translate" title="Перевести">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 8l6 6"/><path d="M4 14l6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="M22 22l-5-10-5 10"/><path d="M14 18h6"/></svg>
            <span>Перевести</span>
        </button>
        <button class="popup-btn" data-action="rewrite" title="Переписать">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            <span>Переписать</span>
        </button>
    `;
    selectionPopup.style.display = 'none';
    document.body.appendChild(selectionPopup);

    function showSelectionPopup(x, y) {
        const selection = window.getSelection();
        if (!selection || selection.toString().trim().length === 0) {
            selectionPopup.style.display = 'none';
            return;
        }
        
        selectionPopup.style.display = 'flex';
        selectionPopup.style.left = x + 'px';
        selectionPopup.style.top = y + 'px';
        
        // Adjust if off-screen
        const rect = selectionPopup.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            selectionPopup.style.left = (window.innerWidth - rect.width - 10) + 'px';
        }
        if (rect.bottom > window.innerHeight) {
            selectionPopup.style.top = (y - rect.height - 10) + 'px';
        }
    }

    function hideSelectionPopup() {
        selectionPopup.style.display = 'none';
    }

    document.addEventListener('mouseup', (e) => {
        const selection = window.getSelection();
        if (selection && selection.toString().trim().length > 0) {
            showSelectionPopup(e.pageX, e.pageY);
        } else {
            // Check if click is on popup
            if (!selectionPopup.contains(e.target)) {
                hideSelectionPopup();
            }
        }
    });

    document.addEventListener('mousedown', (e) => {
        if (!selectionPopup.contains(e.target)) {
            hideSelectionPopup();
        }
    });

    selectionPopup.querySelectorAll('.popup-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const selection = window.getSelection();
            const selectedText = selection ? selection.toString().trim() : '';
            if (!selectedText) {
                hideSelectionPopup();
                return;
            }

            const action = btn.dataset.action;
            
            switch (action) {
                case 'copy':
                    navigator.clipboard.writeText(selectedText).then(() => {
                        toast('Скопировано', 'success');
                    }).catch(() => {
                        toast('Ошибка копирования', 'error');
                    });
                    break;
                    
                case 'ask':
                    $('#chatInput').value = `Расскажи подробнее о: "${selectedText}"`;
                    $('#chatInput').focus();
                    break;
                    
                case 'explain':
                    $('#chatInput').value = `Объясни что значит: "${selectedText}"`;
                    $('#chatInput').focus();
                    break;
                    
                case 'translate':
                    $('#chatInput').value = `Переведи на русский: "${selectedText}"`;
                    $('#chatInput').focus();
                    break;
                    
                case 'rewrite':
                    $('#chatInput').value = `Перепиши это более понятно: "${selectedText}"`;
                    $('#chatInput').focus();
                    break;
            }
            
            hideSelectionPopup();
        });
    });

    // Also hide on scroll
    document.addEventListener('scroll', hideSelectionPopup, true);

    /* ── Analytics ── */
    function loadAnalytics() {
        api('/analytics').then(data => {
            if (data.error) {
                toast('Ошибка загрузки аналитики', 'error');
                return;
            }
            
            $('#analyticsQueries').textContent = data.queries || 0;
            $('#analyticsAvgTime').textContent = (data.avg_time || 0).toFixed(0) + 'ms';
            $('#analyticsSuccess').textContent = ((data.success_rate || 0) * 100).toFixed(0) + '%';
            $('#analyticsCPU').textContent = (data.cpu || 0).toFixed(1) + '%';
            $('#analyticsMemory').textContent = (data.memory || 0).toFixed(1) + '%';
            $('#analyticsVRAM').textContent = (data.vram || 0).toFixed(1) + '%';
            $('#analyticsDialogs').textContent = data.dialogs || 0;
            $('#analyticsGaps').textContent = data.gaps || 0;
            $('#analyticsLearned').textContent = data.learned || 0;
            
            // Render activities
            const activityList = $('#activityList');
            if (data.activities && data.activities.length > 0) {
                activityList.innerHTML = data.activities.map(a => `
                    <div class="activity-item">
                        <div class="activity-icon">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                ${a.icon === 'memory' ? '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>' : 
                                  a.icon === 'learn' ? '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>' :
                                  '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'}
                            </svg>
                        </div>
                        <div class="activity-content">
                            <div class="activity-title">${esc(a.title)}</div>
                            <div class="activity-time">${esc(a.time)}</div>
                        </div>
                    </div>
                `).join('');
            } else {
                activityList.innerHTML = '<div class="empty-state">Нет данных</div>';
            }
        }).catch(() => {
            toast('Ошибка загрузки аналитики', 'error');
        });
    }
    
    $('#refreshAnalytics')?.addEventListener('click', loadAnalytics);

    /* ── Learning ── */
    function loadLearning() {
        api('/learning').then(data => {
            if (data.error) {
                toast('Ошибка загрузки данных обучения', 'error');
                return;
            }
            
            // Update stats
            $('#learnTotal').textContent = data.total || 0;
            $('#learnSuccess').textContent = data.success || 0;
            $('#learnPending').textContent = data.pending || 0;
            
            // Render opportunities
            const oppList = $('#opportunitiesList');
            if (data.opportunities && data.opportunities.length > 0) {
                oppList.innerHTML = data.opportunities.map(o => `
                    <div class="opportunity-item">
                        <div class="opp-header">
                            <span class="opp-concept">${esc(o.concept)}</span>
                            <span class="opp-priority ${o.priority_level}">${(o.priority * 100).toFixed(0)}%</span>
                        </div>
                        <div class="opp-type">${esc(o.type)} • ${esc(o.domain)}</div>
                    </div>
                `).join('');
            } else {
                oppList.innerHTML = '<div class="empty-state">Нет доступных возможностей</div>';
            }
            
            // Render recent dialogs
            const dialogList = $('#dialogsList');
            if (data.recent_dialogs && data.recent_dialogs.length > 0) {
                dialogList.innerHTML = data.recent_dialogs.map(d => `
                    <div class="dialog-item">
                        <div class="dialog-header">
                            <span class="dialog-topic">${esc(d.topic)}</span>
                            <span class="dialog-outcome">${esc(d.outcome)}</span>
                        </div>
                    </div>
                `).join('');
            } else {
                dialogList.innerHTML = '<div class="empty-state">Нет недавних диалогов</div>';
            }
        }).catch(() => {
            toast('Ошибка загрузки обучения', 'error');
        });
    }
    
    $('#refreshLearning')?.addEventListener('click', loadLearning);

    /* ── Settings toggles ── */
    function setupToggle(id, key) {
        const toggle = $(id);
        if (!toggle) return;
        toggle.addEventListener('click', () => {
            toggle.classList.toggle('active');
            const isActive = toggle.classList.contains('active');
            localStorage.setItem('cogniflex_' + key, isActive ? '1' : '0');
            toast(isActive ? 'Включено' : 'Выключено', 'info');
        });
        // Load saved state
        const saved = localStorage.getItem('cogniflex_' + key);
        if (saved === '0') toggle.classList.remove('active');
    }
    
    setupToggle('#toggleAutoLearn', 'auto_learn');
    setupToggle('#toggleSRE', 'sre_enabled');
    setupToggle('#toggleMemory', 'memory_enabled');
    setupToggle('#toggleTheme', 'dark_theme');
    setupToggle('#toggleSound', 'sound_enabled');

})();
