import torch

from text_embedding import OrthogonalCharEmbedding, clean_letters


def test_frozen_deterministic_codebook():
    a = OrthogonalCharEmbedding(vocab_size=4096)
    b = OrthogonalCharEmbedding(vocab_size=4096)
    torch.testing.assert_close(a.encode('باب سلام'), b.encode('باب سلام'), rtol=0, atol=0)
    assert not any(p.requires_grad for p in a.parameters())
    assert torch.equal(a.encode('بب')[0], a.encode('بب')[1])
    assert not torch.equal(a.encode('با')[0], a.encode('با')[1])
    torch.testing.assert_close(a.encode('سلام ').norm(dim=-1), torch.ones(5))
    assert a.embedding.weight[a.PAD_TOKEN_IDX].count_nonzero() == 0
    assert a(torch.tensor([a.PAD_TOKEN_IDX])).count_nonzero() == 0


def test_batch_shape_space_padding_and_cleaning():
    encoder = OrthogonalCharEmbedding(vocab_size=4096)
    assert encoder.encode('سلام').shape == (4,128)
    values, valid = encoder.encode_batch(['سلام','ب '])
    assert values.shape == (2,4,128)
    assert valid.tolist() == [[True]*4,[True,True,False,False]]
    assert values[1,2:].count_nonzero() == 0
    assert encoder.encode('').shape == (0,128)
    assert encoder.encode_batch([])[0].shape == (0,0,128)
    assert ''.join(clean_letters(' سَـلام 123! ﻻ')) == 'سلاملا'
