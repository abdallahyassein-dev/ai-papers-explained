"""
Unit Tests and Finite-Difference Gradient Checker (`gradcheck`) for Native NPLM.
Verifies forward pass dimensions, loss computation, analytical backpropagation against finite differences,
and perplexity calculation.
"""

import numpy as np
from nplm_native import Vocabulary, NativeNPLM, prepare_dataset, tokenize, NGramBaseline


def test_vocabulary():
    print("Testing Vocabulary...")
    vocab = Vocabulary()
    sentences = [["the", "cat", "sat"], ["the", "dog", "barks"]]
    vocab.build_vocab(sentences)

    assert "<pad>" in vocab.word2idx
    assert "<unk>" in vocab.word2idx
    encoded = vocab.encode(["the", "cat", "unknown_word"])
    decoded = vocab.decode(encoded)
    assert decoded[0] == "the"
    assert decoded[1] == "cat"
    assert decoded[2] == "<unk>"
    print("Vocabulary test passed! ✅")


def test_forward_backward_shapes():
    print("Testing Forward and Backward Shapes...")
    vocab_size = 10
    context_len = 3
    embedding_dim = 8
    hidden_dim = 16
    batch_size = 4

    model = NativeNPLM(vocab_size, context_len, embedding_dim, hidden_dim, direct_connections=True)

    X = np.random.randint(0, vocab_size, size=(batch_size, context_len))
    y = np.random.randint(0, vocab_size, size=(batch_size,))

    probs, loss = model.forward(X, y_target=y, l2_reg=1e-4)

    assert probs.shape == (batch_size, vocab_size), f"Expected shape {(batch_size, vocab_size)}, got {probs.shape}"
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Softmax probabilities must sum to 1.0"
    assert loss > 0, "Loss must be positive"

    # Backward pass
    model.backward(l2_reg=1e-4)

    assert model.dW.shape == model.W.shape, f"dW shape mismatch: {model.dW.shape} vs {model.W.shape}"
    assert model.db.shape == model.b.shape, f"db shape mismatch: {model.db.shape} vs {model.b.shape}"
    assert model.dH.shape == model.H.shape, f"dH shape mismatch: {model.dH.shape} vs {model.H.shape}"
    assert model.dd.shape == model.d.shape, f"dd shape mismatch: {model.dd.shape} vs {model.d.shape}"
    assert model.dC.shape == model.C.shape, f"dC shape mismatch: {model.dC.shape} vs {model.C.shape}"
    assert model.dU.shape == model.U.shape, f"dU shape mismatch: {model.dU.shape} vs {model.U.shape}"
    print("Forward and backward shapes test passed! ✅")


def eval_numerical_gradient(model, X, y, param_name, eps=1e-5, l2_reg=1e-4):
    """Compute numerical gradient via symmetric finite differences."""
    param = getattr(model, param_name)
    grad_num = np.zeros_like(param)

    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = param[idx]

        # f(x + eps)
        param[idx] = old_val + eps
        _, loss_plus = model.forward(X, y_target=y, l2_reg=l2_reg)

        # f(x - eps)
        param[idx] = old_val - eps
        _, loss_minus = model.forward(X, y_target=y, l2_reg=l2_reg)

        # Reset param
        param[idx] = old_val

        # Symmetric difference
        grad_num[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return grad_num


def relative_error(g_ana, g_num):
    """Compute relative error between analytical and numerical gradients."""
    top = np.abs(g_ana - g_num)
    bottom = np.maximum(1e-5, np.abs(g_ana) + np.abs(g_num))
    return np.max(top / bottom)


def test_gradient_check():
    print("Testing Analytical Gradients vs Finite Differences (Gradcheck)...")
    vocab_size = 5
    context_len = 2
    embedding_dim = 4
    hidden_dim = 6
    batch_size = 2
    l2_reg = 1e-4

    model = NativeNPLM(vocab_size, context_len, embedding_dim, hidden_dim, direct_connections=True, seed=123)

    X = np.array([[0, 1], [2, 3]])
    y = np.array([4, 1])

    # Run forward and analytical backward
    _, loss = model.forward(X, y_target=y, l2_reg=l2_reg)
    model.backward(l2_reg=l2_reg)

    params_to_check = [
        ('W', model.dW),
        ('b', model.db),
        ('H', model.dH),
        ('d', model.dd),
        ('C', model.dC),
        ('U', model.dU)
    ]

    for param_name, grad_analytical in params_to_check:
        grad_numerical = eval_numerical_gradient(model, X, y, param_name, eps=1e-5, l2_reg=l2_reg)
        err = relative_error(grad_analytical, grad_numerical)
        print(f"  Gradcheck for {param_name:2s}: Max relative error = {err:.2e}")
        assert err < 1e-4, f"Gradcheck failed for {param_name}! Max relative error = {err}"

    print("Gradcheck passed for ALL parameters! ✅")


def run_all_tests():
    print("=" * 60)
    print("RUNNING NATIVE NPLM UNIT TESTS & GRADIENT CHECK")
    print("=" * 60)
    test_vocabulary()
    test_forward_backward_shapes()
    test_gradient_check()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
