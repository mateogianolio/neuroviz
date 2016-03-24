(function (g) {
  'use strict';

  g.start = function (options) {
    var container = document.querySelector(options.container),
        iterations = options.iterations || 100,
        hidden = options.hidden || 4,
        alpha = options.alpha || 1,
        activation = options.layers || [tanh, tanh];

    var X = options.input,
        y = options.output,
        synapses = [],
        layers = [],
        deltas = [],
        k;

    if (!X || !y || X.shape[0] !== y.shape[0] || !container)
      return;

    for (k = 0; k < activation.length; k++) {
      synapses[k] = Matrix.random(
        (k === 0 ? X.shape[1] : hidden),
        (k === activation.length - 1 ? y.shape[1] : hidden),
        2, -1);
    }

    var size = synapses.length;
    (function loop(i) {
      container.dataset.iterations = i;

      // forward propagation
      layers[0] = X
        .multiply(synapses[0])
        .map(activation[0]());

      for (var k = 1; k < size; k++)
        layers[k] = layers[k - 1]
          .multiply(synapses[k])
          .map(activation[k]());

      // backward propagation
      deltas[size - 1] = Matrix
        .subtract(y, layers[size - 1])
        .product(layers[size - 1].map(activation[size - 1](true)));

      for (k = size - 2; k >= 0; k--)
        deltas[k] = deltas[k + 1]
          .multiply(synapses[k + 1].T)
          .product(layers[k].map(activation[k](true)));

      // weight updates
      for (k = size - 1; k > 0; k--)
        synapses[k].add(layers[k - 1].T.multiply(deltas[k]).scale(alpha));
      synapses[0].add(X.T.multiply(deltas[0]).scale(alpha));

      if (--i >= 0)
        requestAnimationFrame(loop.bind(null, i));
      else
        container.dataset.iterations = 'tap to restart';

      renderMatrix(X, { container: options.container, name: 'input' });
      for (k = 0; k < size; k++) {
        if (options.showSynapses)
          renderMatrix(synapses[k], { container: options.container, name: 'syn' + k});
        renderMatrix(layers[k], { container: options.container, name: 'l' + k, activation: activation[k].name });
      }
      renderMatrix(y, { container: options.container, name: 'output' });
      if (options.showDeltas)
        for (k = 0; k < size; k++)
          renderMatrix(deltas[k], { container: options.container, name: '\u0394l' + k});
    }(options.iterations));

    container.addEventListener('click', function () {
      if (container.dataset.iterations === 'tap to restart')
        start(options);
    });
  };

  g.renderMatrix = function (m, options) {
    if (!options.container || !options.name)
      return;

    var r = m.shape[0],
        c = m.shape[1],
        tmp, tr, td, i, j, x;

    var container = document.querySelector(options.container);
    var table =
      container.querySelector('.' + options.name) ||
      document.createElement('table');

    table.className = options.name;
    table.dataset.info = options.name + ' ' + r + '\u00D7' + c;

    if (options.activation)
      table.dataset.activation = options.activation;

    for (i = 0; i < r; i++) {
      tr = table.querySelector('.r' + i) || document.createElement('tr');
      if (!tr.className)
        tr.className = 'r' + i;

      for (j = 0; j < c; j++) {
        x = m.get(i, j);
        td = tr.querySelector('.c' + j) || document.createElement('td');
        if (!td.className)
          td.className = 'c' + j;
        td.style.backgroundColor = color(x);
        td.innerHTML = x;
        td.dataset.x = x;

        if (!td.parentNode)
          tr.appendChild(td);
      }

      if (!tr.parentNode)
        table.appendChild(tr);
    }

    if (!table.parentNode)
      container.appendChild(table);
  };

  g.color = function (x) {
    var t = tanh()(x);
    if (x < -0.9)
      return '#08519c';
    else if (t < -0.7)
      return '#3182bd';
    else if (t < -0.5)
      return '#6baed6';
    else if (t < -0.3)
      return '#bdd7e7';
    else if (t < -0.01)
      return '#eff3ff';
    else if (t < 0.01)
      return '#fff';
    else if (t < 0.3)
      return '#fee5d9';
    else if (t < 0.5)
      return '#fcae91';
    else if (t < 0.7)
      return '#fb6a4a';
    else if (t < 0.9)
      return '#de2d26';
    return '#a50f15';
  };
}(window));
