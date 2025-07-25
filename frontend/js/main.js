// frontend/js/main.js

const visualBuilderForm = document.getElementById('chat-form');
const roleInput = document.getElementById('role-input');
const taskInput = document.getElementById('task-input');
const contextInput = document.getElementById('context-input');
const constraintsInput = document.getElementById('constraints-input');
const visualSendBtn = document.getElementById('send-btn');
const inputArea = document.getElementById('input-area');
const visualModeBtn = document.getElementById('visual-mode-btn');
const guidedModeBtn = document.getElementById('guided-mode-btn');
const chatInner = document.getElementById('chat-inner');
const modelSelector = document.getElementById('model-selector');
const clearChatBtn = document.getElementById('clear-chat-btn');

const API_URL = 'http://127.0.0.1:8000/api/v1/chat';
let conversationHistory = [];
let currentMode = 'guided';

const guidedModeFormHTML = `
  <form id="guided-chat-form" autocomplete="off" class="w-full">
    <div class="relative">
      <input
        id="guided-input"
        type="text"
        class="w-full bg-white px-6 py-4 border-2 border-black rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-black transition"
        placeholder="Type your message or idea here..."
      />
      <button type="submit" id="guided-send-btn" class="absolute right-3 top-1/2 -translate-y-1/2 bg-black text-white rounded-md p-2 hover:bg-neutral-800 transition disabled:bg-gray-400">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14m-7-7l7 7-7 7" /></svg>
      </button>
    </div>
  </form>
`;

// --- THIS IS THE FINAL FIX ---
// We've removed the { behavior: 'smooth' } to make the scroll instant and more reliable.
function autoScroll() {
  const anchor = document.getElementById('chat-anchor');
  if (anchor) {
    anchor.scrollIntoView({ block: 'end' });
  }
}
// --- END OF FIX ---

function setMode(mode) {
  currentMode = mode;
  const existingGuidedForm = document.getElementById('guided-chat-form');
  if (existingGuidedForm) {
    existingGuidedForm.remove();
  }
  if (mode === 'visual') {
    visualBuilderForm.style.display = 'block';
    visualModeBtn.className = 'px-4 py-2 text-sm font-medium text-white bg-black border border-gray-900 rounded-l-lg hover:bg-neutral-800';
    guidedModeBtn.className = 'px-4 py-2 text-sm font-medium text-black bg-white border border-gray-300 rounded-r-md hover:bg-gray-100';
  } else {
    visualBuilderForm.style.display = 'none';
    const modeSwitcher = document.getElementById('mode-switcher');
    modeSwitcher.insertAdjacentHTML('afterend', guidedModeFormHTML);
    guidedModeBtn.className = 'px-4 py-2 text-sm font-medium text-white bg-black border border-gray-900 rounded-r-md hover:bg-neutral-800';
    visualModeBtn.className = 'px-4 py-2 text-sm font-medium text-black bg-white border border-gray-300 rounded-l-lg hover:bg-gray-100';
  }
}

visualModeBtn.addEventListener('click', () => setMode('visual'));
guidedModeBtn.addEventListener('click', () => setMode('guided'));

visualBuilderForm.addEventListener('submit', async function(event) {
  event.preventDefault();
  const userIdea = `
Role: ${roleInput.value.trim()}
Task: ${taskInput.value.trim()}
Context: ${contextInput.value.trim()}
Constraints: ${constraintsInput.value.trim()}
  `.trim();
  if (!userIdea || !taskInput.value.trim()) {
    alert("Please at least describe the main task.");
    return;
  }
  processAndFetch(userIdea);
  roleInput.value = '';
  taskInput.value = '';
  contextInput.value = '';
  constraintsInput.value = '';
  visualSendBtn.disabled = true;
});

inputArea.addEventListener('submit', async function(event) {
    if (event.target.id !== 'guided-chat-form') return;
    event.preventDefault();
    const guidedInput = document.getElementById('guided-input');
    const userMessage = guidedInput.value.trim();
    if (!userMessage) return;
    processAndFetch(userMessage);
    guidedInput.value = '';
});

async function processAndFetch(userContent) {
    if (window.removePlaceholderIfNeeded) window.removePlaceholderIfNeeded();
    displayUserMessage(userContent);
    saveHistory('user', userContent);
    const selectedModel = modelSelector.value;
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: conversationHistory,
                target_model: selectedModel,
                mode: currentMode 
            }),
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const alchemyResponse = await response.json();
        displayAssistantMessage(alchemyResponse.expert_prompt);
        addExplanationTooltip(alchemyResponse.explanation);
        saveHistory('assistant', alchemyResponse);
    } catch (error) {
        console.error("Error fetching from API:", error);
        displayAssistantMessage('Sorry, I ran into a problem.');
    }
}

const allInputs = [roleInput, taskInput, contextInput, constraintsInput];
allInputs.forEach(input => {
    input.addEventListener('input', () => {
        visualSendBtn.disabled = taskInput.value.trim() === '';
    });
});

clearChatBtn.addEventListener('click', () => {
  if (window.clearConversationHistory) window.clearConversationHistory();
});

function displayUserMessage(content) {
  const anchor = document.getElementById('chat-anchor');
  const msg = document.createElement('div');
  msg.className = 'message-bubble user-bubble';
  const pre = document.createElement('pre');
  pre.textContent = content;
  msg.appendChild(pre);
  if(anchor) anchor.before(msg);
  autoScroll();
}

function displayAssistantMessage(content) {
  const anchor = document.getElementById('chat-anchor');
  const msg = document.createElement('div');
  msg.className = 'message-bubble assistant-bubble';
  if (content && content.startsWith('Sorry,')) {
    msg.textContent = content;
  } else {
    const header = document.createElement('div');
    header.className = 'flex justify-between items-center mb-2';
    const title = document.createElement('strong');
    title.textContent = 'Generated Prompt:';
    const copyBtn = document.createElement('button');
    copyBtn.className = 'text-sm font-medium py-1 px-2 rounded-md bg-gray-200 hover:bg-gray-300 transition';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(content).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 2000);
      }).catch(err => { console.error('Failed to copy text: ', err); copyBtn.textContent = 'Error'; });
    });
    header.appendChild(title);
    header.appendChild(copyBtn);
    const pre = document.createElement('pre');
    pre.textContent = content;
    msg.appendChild(header);
    msg.appendChild(pre);
  }
  if(anchor) anchor.before(msg);
  autoScroll();
}

function addExplanationTooltip(text) {
  const anchor = document.getElementById('chat-anchor');
  const wrapper = document.createElement('div');
  wrapper.className = 'tooltip-wrapper';
  wrapper.innerHTML = `<div class="tooltip-icon">?</div><span class="tooltip-text">${text}</span>`;
  if(anchor) anchor.before(wrapper);
  autoScroll();
}

function saveHistory(role, content) {
  conversationHistory.push({ role, content });
  localStorage.setItem('chatHistory', JSON.stringify(conversationHistory));
}

function loadHistory() {
  const saved = localStorage.getItem('chatHistory');
  if (!saved) return;
  conversationHistory = JSON.parse(saved);
  
  if (conversationHistory.length > 0) {
    const placeholder = document.getElementById('placeholder');
    if (placeholder) placeholder.remove();

    for (const msg of conversationHistory) {
      if (msg.role === 'user') {
        displayUserMessage(msg.content);
      } else if (msg.role === 'assistant') {
        if (typeof msg.content === 'object' && msg.content !== null && 'expert_prompt' in msg.content) {
          displayAssistantMessage(msg.content.expert_prompt);
          addExplanationTooltip(msg.content.explanation);
        } else {
          displayAssistantMessage(msg.content);
        }
      }
    }
  }
}

window.clearConversationHistory = function () {
  conversationHistory = [];
  localStorage.removeItem('chatHistory');
  chatInner.innerHTML = `<div id="placeholder" class="w-full flex items-center justify-center min-h-[200px]"><p class="text-gray-400 text-sm text-center">Start typing to generate a prompt ✨</p></div><div id="chat-anchor"></div>`;
};

loadHistory();
setMode('guided');
