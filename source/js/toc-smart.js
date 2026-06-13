(function () {
  function directChildByClass(parent, className) {
    for (var i = 0; i < parent.children.length; i++) {
      if (parent.children[i].classList && parent.children[i].classList.contains(className)) {
        return parent.children[i];
      }
    }
    return null;
  }

  function closestTocItem(node) {
    while (node && node !== document) {
      if (node.classList && node.classList.contains('toc-item')) return node;
      node = node.parentNode;
    }
    return null;
  }

  function setExpanded(item, expanded) {
    var child = directChildByClass(item, 'toc-child');
    var toggle = directChildByClass(item, 'toc-toggle-item');
    if (!child) return;

    item.classList.toggle('toc-expanded', expanded);
    child.style.display = expanded ? 'block' : 'none';

    if (toggle) {
      toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    }
  }

  function expandActiveBranch(content) {
    var activeLink = content.querySelector('.toc-link.active');
    var item = activeLink ? closestTocItem(activeLink) : null;

    content.querySelectorAll('.active-branch').forEach(function (branch) {
      branch.classList.remove('active-branch');
    });

    while (item && content.contains(item)) {
      item.classList.add('active-branch');
      setExpanded(item, true);
      item = closestTocItem(item.parentNode && item.parentNode.parentNode);
    }
  }

  function setupItemRows(content) {
    content.querySelectorAll('.toc-item').forEach(function (item) {
      var child = directChildByClass(item, 'toc-child');
      var link = directChildByClass(item, 'toc-link');
      if (!link || directChildByClass(item, 'toc-row')) return;

      var row = document.createElement('div');
      row.className = 'toc-row';
      item.insertBefore(row, link);

      if (child) {
        item.classList.add('has-toc-child');

        var toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'toc-toggle-item';
        toggle.setAttribute('aria-label', '展开或收起该章节目录');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.innerHTML = '<span aria-hidden="true">▸</span>';

        toggle.addEventListener('click', function (event) {
          event.preventDefault();
          event.stopPropagation();
          setExpanded(item, !item.classList.contains('toc-expanded'));
        });

        row.appendChild(toggle);
        setExpanded(item, false);
      } else {
        var spacer = document.createElement('span');
        spacer.className = 'toc-toggle-spacer';
        spacer.setAttribute('aria-hidden', 'true');
        row.appendChild(spacer);
      }

      row.appendChild(link);
    });
  }

  function setAll(content, expanded) {
    content.classList.toggle('is-expand', expanded);
    content.querySelectorAll('.has-toc-child').forEach(function (item) {
      setExpanded(item, expanded);
    });
  }

  function setupToc() {
    var card = document.querySelector('#card-toc');
    if (!card || card.dataset.smartTocReady === 'true') return;

    var content = card.querySelector('.toc-content');
    var headline = card.querySelector('.item-headline');
    if (!content || !headline) return;

    card.dataset.smartTocReady = 'true';
    card.classList.add('toc-smart');

    setupItemRows(content);
    expandActiveBranch(content);

    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'toc-toggle';
    button.setAttribute('aria-expanded', 'false');
    button.textContent = '全部展开';

    button.addEventListener('click', function () {
      var expanded = !content.classList.contains('is-expand');
      setAll(content, expanded);
      if (!expanded) expandActiveBranch(content);
      button.textContent = expanded ? '全部收起' : '全部展开';
      button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    });

    headline.appendChild(button);

    var syncTimer = null;
    var lastActiveHref = '';

    window.addEventListener('scroll', function () {
      if (content.classList.contains('is-expand') || syncTimer) return;

      syncTimer = window.setTimeout(function () {
        syncTimer = null;
        var activeLink = content.querySelector('.toc-link.active');
        var activeHref = activeLink ? activeLink.getAttribute('href') : '';

        if (activeHref && activeHref !== lastActiveHref) {
          lastActiveHref = activeHref;
          expandActiveBranch(content);
        }
      }, 180);
    }, { passive: true });
  }

  document.addEventListener('DOMContentLoaded', setupToc);
  document.addEventListener('pjax:complete', setupToc);
  window.addEventListener('load', setupToc);
})();
