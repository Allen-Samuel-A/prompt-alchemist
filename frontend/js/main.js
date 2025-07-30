// Updated main.js with UI/UX improvements

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
const spinner = document.getElementById('loading-spinner');
const helpBtn = document.getElementById('help-btn');
const helpDrawer = document.getElementById('help-drawer');

const API_URL = 'https://prompt-alchemist.onrender.com/api/v1/chat';
let conversationHistory = [];
let currentMode = 'guided';

window.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('privacy-modal');
  const closeBtn = document.getElementById('close-modal-btn');
  const dismissCheckbox = document.getElementById('dismiss-forever');

  if (!localStorage.getItem('privacyDismissed')) {
    modal.classList.remove('hidden');
  }

  closeBtn.addEventListener('click', () => {
    if (dismissCheckbox.checked) {
      localStorage.setItem('privacyDismissed', 'true');
    }
    modal.classList.add('hidden');
  });

  helpBtn.addEventListener('click', () => {
    helpDrawer.classList.toggle('hidden');
  });
});

function autoScroll() {
  const anchor = document.getElementById('chat-anchor');
  if (anchor) anchor.scrollIntoView({ block: 'end' });
}

function setMode(mode) {
  currentMode = mode;
  localStorage.setItem('lastMode', mode);
  const existingGuidedForm = document.getElementById('guided-chat-form');
  if (existingGuidedForm) existingGuidedForm.remove();

  if (mode === 'visual') {
    visualBuilderForm.style.display = 'block';
    visualModeBtn.classList.add('bg-black', 'text-white');
    guidedModeBtn.classList.remove('bg-black', 'text-white');
  } else {
    visualBuilderForm.style.display = 'none';
    const modeSwitcher = document.getElementById('mode-switcher');
    modeSwitcher.insertAdjacentHTML('afterend', guidedModeFormHTML);
    guidedModeBtn.classList.add('bg-black', 'text-white');
    visualModeBtn.classList.remove('bg-black', 'text-white');
  }
}

const guidedModeFormHTML = `
  <form id="guided-chat-form" autocomplete="off" class="w-full mt-4">
    <div class="relative">
      <input id="guided-input" type="text" class="w-full bg-white px-6 py-4 border-2 border-black rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-black transition" placeholder="Type your message or idea here..." />
      <button type="submit" id="guided-send-btn" class="absolute right-3 top-1/2 -translate-y-1/2 bg-black text-white rounded-md p-2 hover:bg-neutral-800 transition disabled:bg-gray-400">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14m-7-7l7 7-7 7" />
        </svg>
      </button>
    </div>
  </form>
`;

visualModeBtn.addEventListener('click', () => setMode('visual'));
guidedModeBtn.addEventListener('click', () => setMode('guided'));

visualBuilderForm.addEventListener('submit', async function (event) {
  event.preventDefault();
  const userIdea = `
Role: ${roleInput.value.trim()}
Task: ${taskInput.value.trim()}
Context: ${contextInput.value.trim()}
Constraints: ${constraintsInput.value.trim()}`.trim();

  if (!userIdea || !taskInput.value.trim()) {
    alert('Please at least describe the main task.');
    return;
  }

  // Clear inputs immediately for better UX
  roleInput.value = '';
  taskInput.value = '';
  contextInput.value = '';
  constraintsInput.value = '';

  visualSendBtn.disabled = true;
  spinner.classList.remove('hidden');
  await processAndFetch(userIdea);
  spinner.classList.add('hidden');
  visualSendBtn.disabled = false;
});

inputArea.addEventListener('submit', async function (event) {
  if (event.target.id !== 'guided-chat-form') return;
  event.preventDefault();
  const guidedInput = document.getElementById('guided-input');
  const userMessage = guidedInput.value.trim();
  if (!userMessage) return;
  guidedInput.value = '';
  spinner.classList.remove('hidden');
  await processAndFetch(userMessage);
  spinner.classList.add('hidden');
});

async function processAndFetch(userContent) {
  if (window.removePlaceholderIfNeeded) window.removePlaceholderIfNeeded();
  displayUserMessage(userContent);
  saveHistory('user', userContent);
  const selectedModel = modelSelector.value;

  const typingBubble = document.createElement('div');
  typingBubble.className = 'message-bubble assistant-bubble typing-indicator';
  typingBubble.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  chatInner.insertBefore(typingBubble, document.getElementById('chat-anchor'));
  autoScroll();

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: conversationHistory, target_model: selectedModel, mode: currentMode })
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const alchemyResponse = await response.json();
    typingBubble.remove();
    displayAssistantMessage(alchemyResponse.expert_prompt, alchemyResponse.explanation);
    saveHistory('assistant', alchemyResponse);
  } catch (error) {
    console.error('Error fetching from API:', error);
    typingBubble.remove();
    displayAssistantMessage('Sorry, I ran into a problem.');
  }
}

function displayUserMessage(content) {
  const anchor = document.getElementById('chat-anchor');
  const msg = document.createElement('div');
  msg.className = 'message-bubble user-bubble';
  const pre = document.createElement('pre');
  pre.textContent = content;
  msg.appendChild(pre);
  msg.appendChild(buildActions(content));
  if (anchor) anchor.before(msg);
  autoScroll();
}

function displayAssistantMessage(content, explanation) {
  const anchor = document.getElementById('chat-anchor');
  const msg = document.createElement('div');
  msg.className = 'message-bubble assistant-bubble';

  const container = document.createElement('div');
  container.className = 'space-y-2';

  const header = document.createElement('div');
  header.className = 'flex justify-between items-center';
  const title = document.createElement('strong');
  title.textContent = '💡 AI Suggestion:';
  header.appendChild(title);
  container.appendChild(header);

  const pre = document.createElement('pre');
  pre.textContent = content;
  container.appendChild(pre);

  if (explanation) {
    const expBtn = document.createElement('button');
    expBtn.className = 'text-sm underline text-blue-600';
    expBtn.textContent = 'View Explanation';
    const expBox = document.createElement('div');
    expBox.className = 'text-sm mt-2 p-3 border rounded bg-gray-50 hidden';
    expBox.textContent = explanation;
    expBtn.onclick = () => expBox.classList.toggle('hidden');
    container.appendChild(expBtn);
    container.appendChild(expBox);
  }

  msg.appendChild(container);
  msg.appendChild(buildActions(content));
  if (anchor) anchor.before(msg);
  autoScroll();
}

function buildActions(text) {
  const actions = document.createElement('div');
  actions.className = 'text-xs text-gray-500 mt-1 flex gap-4';
  actions.innerHTML = `
    <button onclick="navigator.clipboard.writeText(\`${text}\`)">📋 Copy</button>
    <button onclick="this.closest('.message-bubble').remove()">🗑️ Delete</button>
    <button onclick="alert('Favorited! (Placeholder)')">⭐ Favorite</button>
  `;
  return actions;
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
      if (msg.role === 'user') displayUserMessage(msg.content);
      else if (msg.role === 'assistant') {
        if (typeof msg.content === 'object') {
          displayAssistantMessage(msg.content.expert_prompt, msg.content.explanation);
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
  chatInner.innerHTML = `<div id="placeholder" class="w-full flex items-center justify-center min-h-[200px]">
    <p class="text-gray-400 text-sm text-center">Start typing to generate a prompt ✨</p>
  </div><div id="chat-anchor"></div>`;
};

const allInputs = [roleInput, taskInput, contextInput, constraintsInput];
allInputs.forEach(input => {
  input.addEventListener('input', () => {
    visualSendBtn.disabled = taskInput.value.trim() === '';
  });
});

clearChatBtn.addEventListener('click', () => {
  if (window.clearConversationHistory) window.clearConversationHistory();
});

loadHistory();
setMode(localStorage.getItem('lastMode') || 'guided');
