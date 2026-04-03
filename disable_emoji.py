with open('C:/Users/black/OneDrive/Desktop/CogniFlex/eva/gui/web_gui/templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Disable emoji button - add style and disabled attribute
old_block = '''                        <button class="emoji-btn" id="emojiBtn" title="Эмодзи">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
                        </button>'''

new_block = '''                        <button class="emoji-btn" id="emojiBtn" title="Эмодзи (отключено)" style="opacity: 0.3; pointer-events: none;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
                        </button>'''

content = content.replace(old_block, new_block)

with open('C:/Users/black/OneDrive/Desktop/CogniFlex/eva/gui/web_gui/templates/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done - Emoji button disabled')