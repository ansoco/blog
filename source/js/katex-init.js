document.addEventListener("DOMContentLoaded", function() {
  // 处理 Pandoc 生成的数学公式
  var mathElements = document.querySelectorAll('.math.inline, .math.display');
  mathElements.forEach(function(el) {
    var tex = el.textContent || el.innerText;
    var isDisplay = el.classList.contains('display');
    
    try {
      katex.render(tex, el, {
        displayMode: isDisplay,
        throwOnError: false,
        trust: true,
        strict: false
      });
    } catch (e) {
      console.warn('KaTeX render error:', e);
    }
  });
  
  // 同时处理传统的 $$ 和 $ 格式（作为后备）
  if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "$", right: "$", display: false},
        {left: "\\[", right: "\\]", display: true},
        {left: "\\(", right: "\\)", display: false}
      ],
      throwOnError: false,
      trust: true,
      strict: false
    });
  }
});
