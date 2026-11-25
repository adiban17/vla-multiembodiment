"""Phase 2 checks: 10 paraphrases per task, 8 for training and 2 held out."""

from vla.data.instructions import ROBOTS, all_instructions, instruction_id, paraphrases


def test_ten_unique_paraphrases_per_task_split_8_2():
    for robot in ROBOTS:
        train, heldout = paraphrases(robot, "train"), paraphrases(robot, "heldout")
        assert len(train) == 8 and len(heldout) == 2
        assert len(set(train + heldout)) == 10
        assert not set(train) & set(heldout)


def test_global_index_is_stable():
    sentences = all_instructions()
    assert len(sentences) == 20 and len(set(sentences)) == 20
    for robot in ROBOTS:
        for k, text in enumerate(paraphrases(robot, "all")):
            assert sentences[instruction_id(robot, k)] == text
    assert instruction_id("pusht", 0) == 0 and instruction_id("fetch", 0) == 10
