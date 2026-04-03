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
        
        // Build URL with query params if provided
        let url = `/api${path}`;
        if (opts.params) {
            const params = new URLSearchParams(opts.params);
            url += '?' + params.toString();
            delete opts.params;
        }
        
        const r = await fetch(url, {
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
            // Check if reasoning is structured (array of steps) or plain text
            if (Array.isArray(reasoning) && reasoning.length > 0) {
                // Render structured reasoning steps
                const stepsHtml = reasoning.map((step, idx) => {
                    const phase = step.phase || step.action || 'unknown';
                    const thought = step.thought || step.text || '';
                    const conf = step.confidence || 0;
                    const confClass = conf >= 0.7 ? 'high' : conf >= 0.4 ? 'medium' : 'low';
                    const confLabel = (conf * 100).toFixed(0) + '%';
                    const icons = {
                        'generation': '💭',
                        'model_a_generation': '🧠',
                        'model_b_generation': '💡',
                        'query_analysis': '🔍',
                        'quality_check_a': '✅',
                        'quality_check_b': '✅',
                        'final_synthesis': '📝',
                        'contradiction_check': '⚖️',
                        'ethics_check': '🛡️',
                        'web_search': '🌐',
                        'refinement': '✨'
                    };
                    const icon = icons[phase] || '🔹';
                    return `
                        <div class="reasoning-step">
                            <span class="step-icon">${icon}</span>
                            <span class="step-num">${idx + 1}</span>
                            <span class="step-phase">${esc(phase)}</span>
                            <span class="step-thought">${esc(thought)}</span>
                            <span class="step-conf ${confClass}">${confLabel}</span>
                        </div>
                    `;
                }).join('');
                reasoningHtml = `
                    <div class="msg-reasoning">
                        <button class="reasoning-toggle" onclick="this.classList.toggle('open')">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                            Рассуждения (${reasoning.length} шагов)
                        </button>
                        <div class="reasoning-body">
                            <div class="reasoning-steps">${stepsHtml}</div>
                        </div>
                    </div>
                `;
            } else {
                // Plain text reasoning
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
            }
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

        // Add action buttons for all messages (copy, like, dislike, regenerate)
        const actionsHtml = `
            <div class="msg-actions">
                <button class="msg-action-btn" onclick="copyMessage(this, \`${esc(text)}\`)" title="Копировать">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    Копировать
                </button>
                <button class="msg-action-btn like" onclick="rateMessage('\`${esc(text)}\`', 1)" title="Полезно">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
                    Полезно
                </button>
                <button class="msg-action-btn dislike" onclick="rateMessage('\`${esc(text)}\`', -1)" title="Неверно">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>
                    Неверно
                </button>
                ${role !== 'user' ? `
                <button class="msg-action-btn regenerate" onclick="regenerateMessage(this, \`${esc(text)}\`)" title="Перегенерировать">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/></svg>
                    Перегенерировать
                </button>
                ` : ''}
            </div>
        `;

        div.innerHTML = `
            <div class="msg-inner">
                <div class="msg-role ${roleClass}">${roleLabel}</div>
                <div class="msg-text">${formatText(text)}</div>
                ${fileHtml}
                ${reasoningHtml}
                ${actionsHtml}
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

    /* ── Documents ── */
    function loadDocuments() {
        if (!activeSessionId) return;
        
        api('/documents', { params: { session_id: activeSessionId } }).then(data => {
            if (data.error) return;
            
            const docs = data.documents || [];
            const docEl = $('#documentList');
            
            if (docEl) {
                if (docs.length > 0) {
                    docEl.innerHTML = docs.map(d => `
                        <div class="doc-item">
                            <div class="doc-icon">📄</div>
                            <div class="doc-info">
                                <div class="doc-name">${esc(d.filename || 'Unknown')}</div>
                                <div class="doc-meta">${d.doc_type || 'file'} · ${esc(d.file_id?.substring(0, 8) || '')}</div>
                            </div>
                        </div>
                    `).join('');
                } else {
                    docEl.innerHTML = '<div class="empty-state">Нет загруженных документов</div>';
                }
            }
        }).catch(() => {});
    }
    
    /* ── Knowledge Graph ── */
    function loadKnowledgeGraph() {
        api('/knowledge-graph', { params: { action: 'get' } }).then(data => {
            if (data.error) return;
            
            const nodes = data.nodes || [];
            const kgEl = $('#knowledgeGraph');
            
            if (kgEl) {
                if (nodes.length > 0) {
                    kgEl.innerHTML = nodes.map(n => `
                        <div class="kg-node">
                            <div class="kg-node-name">${esc(n.name || n.id)}</div>
                            <div class="kg-node-content">${esc(n.content || '')}</div>
                        </div>
                    `).join('');
                } else {
                    kgEl.innerHTML = '<div class="empty-state">Граф знаний пуст</div>';
                }
            }
        }).catch(() => {});
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
    
    $('#refreshMemory')?.addEventListener('click', () => {
        loadMemory();
        loadDocuments();
        loadKnowledgeGraph();
    });

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
            
            // Рассуждения - передаём structured steps для рендеринга карточками
            let reasoningData = null;

            // 1. Основные рассуждения (steps) - передаём как массив
            if (d.reasoning_steps && d.reasoning_steps.length > 0) {
                reasoningData = d.reasoning_steps;
            }

            // 2. Веб-поиск - добавляем как дополнительный шаг
            if (d.web_search_info) {
                if (!reasoningData) reasoningData = [];
                reasoningData.push({
                    step: reasoningData.length + 1,
                    phase: 'web_search',
                    thought: d.web_search_info,
                    confidence: 0.8
                });
            }

            // 3. Самодиалог - добавляем как дополнительный шаг
            if (d.self_dialog) {
                const sd = d.self_dialog;
                let sdText = 'Самодиалог: ' + (sd.topic || '');
                if (sd.outcome) sdText += ' → ' + sd.outcome;
                if (sd.gaps && sd.gaps.length > 0) sdText += '. Пробелы: ' + sd.gaps.join(', ');
                if (!reasoningData) reasoningData = [];
                reasoningData.push({
                    step: reasoningData.length + 1,
                    phase: 'self_dialog',
                    thought: sdText,
                    confidence: sd.confidence || 0.5
                });
            }

            // Fallback на plain text reasoning
            if (!reasoningData && d.reasoning) {
                reasoningData = d.reasoning;
            }

            // Добавляем финальный ответ со структурированными рассуждениями
            addMsg('system', d.response || 'Нет ответа', null, reasoningData);
            
            // Показываем уточняющий вопрос если есть
            if (d.clarification_question) {
                const clarHtml = `
                    <div class="clarification-box">
                        <div class="clarification-icon">❓</div>
                        <div class="clarification-text">${esc(d.clarification_question)}</div>
                    </div>
                `;
                const chatContainer = document.getElementById('chatMessages');
            const lastMsg = chatContainer ? chatContainer.lastElementChild : null;
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

    /* ── System Info ── */
    function loadSystemInfo() {
        api('/system').then(data => {
            if (data.error) return;
            
            // Update model in settings
            const modelEl = document.querySelector('#settingsView .setting-group:last-child .setting-row:nth-child(2) span');
            if (modelEl) {
                modelEl.textContent = data.model;
            }
            
            // Update status indicator
            const statusDot = $('.status-dot');
            const statusText = $('.status-text');
            
            if (statusDot && statusText) {
                if (data.llama_cpp_ready || data.qwen_ready) {
                    statusDot.classList.add('active');
                    statusText.textContent = 'Готов';
                } else {
                    statusDot.classList.remove('active');
                    statusText.textContent = 'Загрузка...';
                }
            }
        }).catch(() => {});
    }
    
    // Check system status on load and periodically
    loadSystemInfo();
    setInterval(loadSystemInfo, 30000);
    
    /* ── Cache Stats ── */
    function loadCacheStats() {
        api('/cache-stats').then(data => {
            if (data.error) return;
            
            // Could display cache stats somewhere if needed
            console.log('Cache stats:', data);
        }).catch(() => {});
    }
    
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

    /* ── Model Status ── */
    function loadModelStatus() {
        api('/model-status').then(data => {
            if (data.error) return;
            
            // Update model indicators
            const models = data.models || {};
            Object.keys(models).forEach(key => {
                const model = models[key];
                const el = $(`#model-${key}`);
                if (el) {
                    el.className = `model-indicator ${model.loaded ? 'loaded' : 'unloaded'}`;
                    el.innerHTML = `
                        <span class="model-name">${model.name}</span>
                        <span class="model-role">${model.role}</span>
                        <span class="model-status">${model.loaded ? 'Загружена' : 'Не загружена'}</span>
                    `;
                }
            });
            
            // Update pipeline status
            const pipelineEl = $('#pipelineStatus');
            if (pipelineEl) {
                pipelineEl.className = `pipeline-status ${data.pipeline_ready ? 'ready' : 'not-ready'}`;
                pipelineEl.textContent = data.pipeline_ready ? 'Пайплайн готов' : 'Пайплайн не готов';
            }
            
            // Update fractal memory info
            const fm = data.fractal_memory || {};
            const fmEl = $('#fractalMemoryInfo');
            if (fmEl && fm.enabled) {
                fmEl.innerHTML = `
                    <span>Узлов: ${fm.nodes}</span>
                    <span>Связей: ${fm.edges}</span>
                    <span>Опытов: ${fm.experiences}</span>
                    <span>Концептов: ${fm.concepts}</span>
                `;
            }
        }).catch(() => {});
    }
    
    // Load model status on startup
    if ($('#modelStatus')) loadModelStatus();
    setInterval(loadModelStatus, 60000);

    /* ── Self-Dialog ── */
    function loadSelfDialog() {
        api('/self-dialog').then(data => {
            if (data.error) return;
            
            const statusEl = $('#selfDialogStatus');
            if (statusEl) {
                statusEl.innerHTML = `
                    <div class="sd-stat">
                        <span class="sd-label">Всего диалогов</span>
                        <span class="sd-value">${data.total_dialogs || 0}</span>
                    </div>
                    <div class="sd-stat">
                        <span class="sd-label">Успешных</span>
                        <span class="sd-value">${data.successful || 0}</span>
                    </div>
                    <div class="sd-stat">
                        <span class="sd-label">Неудачных</span>
                        <span class="sd-value">${data.failed || 0}</span>
                    </div>
                `;
            }
        }).catch(() => {});
    }
    
    // Trigger self-dialog
    function triggerSelfDialog() {
        api('/self-dialog', {
            method: 'POST',
            body: {}
        }).then(data => {
            if (data.status === 'success') {
                toast('Самодиалог запущен: ' + (data.topic || 'общая тема'), 'success');
                loadSelfDialog();
            } else {
                toast('Ошибка запуска самодиалога: ' + (data.error || ''), 'error');
            }
        }).catch(() => {
            toast('Ошибка запуска самодиалога', 'error');
        });
    }
    
    $('#triggerSelfDialog')?.addEventListener('click', triggerSelfDialog);
    if ($('#selfDialogStatus')) loadSelfDialog();

    /* ── Snapshots ── */
    function loadSnapshots() {
        api('/snapshots').then(data => {
            if (data.error) return;
            
            const listEl = $('#snapshotList');
            if (listEl && data.snapshots) {
                if (data.snapshots.length > 0) {
                    listEl.innerHTML = data.snapshots.map(s => `
                        <div class="snapshot-item">
                            <span class="snapshot-name">${esc(s.name)}</span>
                            <span class="snapshot-info">${s.experiences} опытов, ${s.concepts} концептов</span>
                        </div>
                    `).join('');
                } else {
                    listEl.innerHTML = '<div class="empty-state">Нет слепков</div>';
                }
            }
        }).catch(() => {});
    }
    
    function createSnapshot() {
        api('/snapshots', {
            method: 'POST',
            body: {}
        }).then(data => {
            if (data.status === 'success') {
                toast('Слепок создан', 'success');
                loadSnapshots();
            } else {
                toast('Ошибка создания слепка', 'error');
            }
        }).catch(() => {
            toast('Ошибка создания слепка', 'error');
        });
    }
    
    $('#createSnapshot')?.addEventListener('click', createSnapshot);
    if ($('#snapshotList')) loadSnapshots();

    /* ── Message Actions ── */
    function copyMessage(btn, text) {
        navigator.clipboard.writeText(text).then(() => {
            toast('Скопировано', 'success');
        }).catch(() => {
            toast('Ошибка копирования', 'error');
        });
    }
    
    function rateMessage(text, rating) {
        console.log('Rate message:', rating, text);
        toast(rating === 1 ? 'Спасибо за оценку!' : 'О учтём', 'success');
    }
    
    function regenerateMessage(btn, oldText) {
        const msgDiv = btn.closest('.msg');
        if (!msgDiv) return;
        
        const userMsgDiv = msgDiv.previousElementSibling;
        if (!userMsgDiv || !userMsgDiv.classList.contains('user')) {
            toast('Не найден предыдущий запрос', 'error');
            return;
        }
        
        const userText = userMsgDiv.querySelector('.msg-text')?.textContent;
        if (!userText) {
            toast('Не удалось получить текст запроса', 'error');
            return;
        }
        
        msgDiv.remove();
        sendMessage(userText);
    }

})();