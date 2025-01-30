import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tests.setup_tests.setup_modelling_test_data.setup_targets_test_data import (
    get_current_test_label_values,
    get_test_label_file_fieldnames,
    set_up_label_file_writing,
    set_up_label_line_dict,
)
from tests.setup_tests.setup_modelling_test_data.setup_test_data_utils import (
    set_up_test_data_root_outpath,
)

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_data import TestDataConfig


def create_test_sequence_data(
    test_data_config: "TestDataConfig", sequence_outfolder: Path
) -> Path:
    c = test_data_config

    fieldnames = get_test_label_file_fieldnames()
    label_file_handle, label_file_writer = set_up_label_file_writing(
        base_path=c.scoped_tmp_path, fieldnames=fieldnames, extra_name="_sequence"
    )

    sequence_outfolder = set_up_test_data_root_outpath(base_folder=sequence_outfolder)
    test_sequence_label_keywords = get_continent_keyword_map()
    test_sequence_random_pool = tuple(get_text_sequence_base())

    samples_for_csv = []
    for cls, _snp_row_idx in c.target_classes.items():
        for sample_idx in range(c.n_per_class):
            sample_outpath = sequence_outfolder / f"{sample_idx}_{cls}.txt"

            cur_sequence = _create_test_sequence(
                min_length=16,
                max_length=64,
                random_pool=test_sequence_random_pool,
                target_keywords_to_include=test_sequence_label_keywords[cls],
                target_class=cls,
            )
            sample_outpath.write_text(data=cur_sequence.sequence)
            samples_for_csv.append(
                {
                    "ID": sample_outpath.stem,
                    "Sequence": cur_sequence.sequence,
                }
            )

            label_line_base = set_up_label_line_dict(
                sample_name=sample_outpath.stem, fieldnames=fieldnames
            )

            label_line_dict = get_current_test_label_values(
                values_dict=label_line_base,
                num_active_elements_in_sample=len(
                    cur_sequence.target_class_keywords_in_sequence
                ),
                cur_class=cls,
            )
            label_file_writer.writerow(label_line_dict)

    label_file_handle.close()
    df_sequence = pd.DataFrame(data=samples_for_csv)
    df_sequence.to_csv(path_or_buf=c.scoped_tmp_path / "sequence.csv", index=False)

    return sequence_outfolder


@dataclass
class SequenceTestSample:
    sequence: str
    target_class: str
    target_class_keywords_in_sequence: Sequence[str]


def _create_test_sequence(
    min_length: int,
    max_length: int,
    random_pool: Sequence[str],
    target_keywords_to_include: Sequence[str],
    target_class: str,
    min_targets_to_include: int = 5,
    max_targets_to_include: int = 20,
) -> SequenceTestSample:
    seq_length = random.choice(seq=range(min_length, max_length))

    test_sequence = random.choices(population=random_pool, k=seq_length)

    index_target_candidates = list(range(seq_length))
    num_target_keywords_to_use = random.choice(
        seq=range(min_targets_to_include, max_targets_to_include)
    )
    num_target_keywords_to_use = min(seq_length, num_target_keywords_to_use)
    target_indices = random.sample(
        population=index_target_candidates, k=num_target_keywords_to_use
    )

    target_keywords = []
    for index in target_indices:
        target_keyword = random.choice(target_keywords_to_include)
        test_sequence[index] = target_keyword
        target_keywords.append(target_keyword)

    seq_test_sample = SequenceTestSample(
        sequence=" ".join(test_sequence),
        target_class=target_class,
        target_class_keywords_in_sequence=target_keywords,
    )

    return seq_test_sample


def get_continent_keyword_map() -> dict[str, Sequence[str]]:
    map_ = {
        "Africa": [
            "giraffe",
            "elephant",
            "gorilla",
            "zebra",
            "hyena",
            "rhino",
            "kudu",
            "lion",
        ],
        "Asia": [
            "tiger",
            "panda",
            "leopard",
            "bengal",
            "pangolin",
            "komodo",
            "cobra",
        ],
        "Europe": [
            "moose",
            "polar_bear",
            "lynx",
            "reindeer",
            "arctic_fox",
            "sheep",
            "mink",
        ],
    }

    return map_


def get_text_sequence_base() -> set:
    base = (
        "disarm explosives arsenal battles ninja houses decoded ai-controlled "
        "patriot seemingly along titled order raised mercenaries exchange disc "
        "himself truth discloses lover submersible orchestrated seeing foxhound "
        "raiden oil purpose with could leader finds epilogue captured are "
        "fortress russian dies up tells while she who country having encounters "
        "par hunt to liberian via frees status stabs replicate ordered snake "
        "become injury aid the rescues peter an cleanup overthrow responding "
        "cell transport step-sister environmental threaten as again replaced "
        "find take united emma upon eliminate bomb sons rose right allegedly "
        "parents twelve explains solidus capture protect drowning mysterious "
        "prematurely identifying then raid patriots on it disable impersonating "
        "reunited hostages war acting society george computer destroy double-agent "
        "collapses knowledge big was ai s3 planted otacon metal wiseman known "
        "timed and fatman own stolen locate will civil furthermore tour plan "
        "clear uploads control called top erected manhattan digital reveals "
        "virus president ocelot during attacked carrying era taken gain mobile "
        "years arm pregnant liquid has iroquois forced sham information hostage "
        "spill johnson gurlukovich explaining he navy ship seize states gw "
        "child malfunction boss committee battle be so after into when have "
        "soldier murdered olga confronted facade later solid predecessor affair "
        "flying entire members shot him destroyed colonel they clone gear "
        "possessed whose kills entry prevent however regression pliskin cut by "
        "is cyborg stillman team highest shoots his damaged fortune helicopter "
        "in infiltrates james sears model facility killed rays scuttles but "
        "arrange seal helped result spy real tracking data led severed emmerich "
        "plans hide being agent of artificially shell for safety escapes human "
        "host awakens council trivial been surviving staged crashes begins new "
        "loses all democratic only using discovers vamp betrays rescue before "
        "helps organization 2007 liberty help us hidden causes harrier a down "
        "ray leaves process upload off dead details appears erratically "
        "construct revealed terrorists departs rule programmer were tanker "
        "had her secretly thought refusal goes pursues hundred from killing "
        "contains that daughter allies contacted deaths defeats two"
    )
    base_set = set(base.split())

    return base_set
