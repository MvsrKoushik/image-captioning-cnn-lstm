from image_captioning import Vocabulary


def test_round_trip_and_unknown_token():
    vocab = Vocabulary(["A red bird", "A blue bird"], minimum_frequency=1)
    assert vocab.decode(vocab.encode("a red bird")) == "a red bird"
    assert vocab.stoi["<unk>"] in vocab.encode("unseen")

