// Service worker — stores last analysis result per tab in memory.
const _results = {};

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'RESULT') {
    const tabId = sender.tab?.id;
    if (tabId != null) {
      _results[tabId] = { result: msg.result, timestamp: Date.now() };
    }
    return;
  }

  if (msg.type === 'GET_RESULT') {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tabId = tabs[0]?.id;
      sendResponse(tabId != null ? (_results[tabId] ?? null) : null);
    });
    return true; // keep message channel open for async response
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  delete _results[tabId];
});
