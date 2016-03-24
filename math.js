(function (g) {
  'use strict';

  g.minimum = function (a, b) {
    return a < b ? a : b;
  };

  g.maximum = function (a, b) {
    return a < b ? b : a;
  };

  g.sum = function (a, b) {
    return a + b;
  };

  g.sinh = function sinh(x) {
    var p = Math.exp(x),
        n = 1 / p;
    return (p - n) / 2;
  };

  g.cosh = function cosh(x) {
    var p = Math.exp(x),
        n = 1 / p;
    return (p + n) / 2;
  };

  // f(x) = tanh(x)
  // f'(x) = 1 - f(x)^2
  g.tanh = function tanh(ddx) {
    return function (x) {
      return ddx ?
        1 - Math.pow(x, 2) :
        sinh(x) / cosh(x);
    };
  };

  // f(x) = x
  // f'(x) = 1
  g.identity = function identity(ddx) {
    return function (x) {
      return ddx ?
        1 :
        x;
    };
  };

  // f(x) = x < 0 ? 0 : x
  // f'(x) = x < 0 ? 0 : 1
  g.relu = function relu(ddx) {
    return function (x) {
      return ddx ?
        x < 0 ? 0 : 1 :
        x < 0 ? 0 : x;
    };
  };

  // f(x) = 1 / (1 + e^(-x))
  // f'(x) = f(x) * (1 - f(x))
  g.sigmoid = function sigmoid(ddx) {
    return function (x) {
      return ddx ?
        x * (1 - x) :
        1 / (1 + Math.exp(-x));
    };
  };

  // f(x) = log(1 + e^x)
  // f'(x) = 1 / (1 + e^(-x))
  g.softplus = function softplus(ddx) {
    return function (x) {
      return ddx ?
        sigmoid()(x) :
        Math.log(1 + Math.exp(x));
    };
  };

  g.softmax = function softmax(ddx) {
    return function (x, i, j) {
      var c = this.shape[1],
          z = Math.exp(x),
          sum = 0;
      for (var k = 0; k < c; k++)
        sum += Math.exp(this.get(i, k));

      z /= sum;
      return ddx ?
        sigmoid(true)(z) :
        z;
    };
  };
}(window));
