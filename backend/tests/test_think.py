from app.llm.think import ThinkStripper


def test_strips_complete_think_block():
    stripper = ThinkStripper()
    visible = stripper.feed("<think>secret plan</think>\nREADY")
    visible += stripper.flush()
    assert "secret" not in visible
    assert "READY" in visible


def test_strips_think_split_across_chunks():
    stripper = ThinkStripper()
    out = []
    for piece in ["<th", "ink>hidden</th", "ink>\nHello"]:
        out.append(stripper.feed(piece))
    out.append(stripper.flush())
    text = "".join(out)
    assert "hidden" not in text
    assert "Hello" in text
