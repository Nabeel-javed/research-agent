from app.ingest.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("notes.txt", "   \n", 800, 150) == []


def test_short_text_is_one_chunk():
    chunks = chunk_text("notes.txt", "BoWatt builds energy software.", 800, 150)
    assert len(chunks) == 1
    assert chunks[0].file_name == "notes.txt"
    assert "BoWatt" in chunks[0].text


def test_long_text_overlaps():
    text = " ".join(f"token-{index:03d}" for index in range(120))
    chunks = chunk_text("big.txt", text, chunk_size=160, overlap=20)
    assert len(chunks) > 1
    assert chunks[0].index == 0
    assert chunks[1].index == 1
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())
    for token in text.split():
        assert any(token in chunk.text for chunk in chunks)


def test_prefers_structural_boundaries():
    text = "Summary line.\n\nExperience\nNorthstar\nCurrent role details continue here."
    chunks = chunk_text("resume.txt", text, chunk_size=35, overlap=3)
    assert any("Experience" in chunk.text and "Northstar" in chunk.text for chunk in chunks)
    for word in text.split():
        assert any(word in chunk.text for chunk in chunks)


def test_rejects_invalid_overlap():
    try:
        chunk_text("notes.txt", "content", chunk_size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlap equal to chunk_size must be rejected")
