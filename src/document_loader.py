"""Load the banking FAQ CSV and turn each row into a document."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config import DATASET_PATH, logger

REQUIRED_COLUMNS: tuple[str, ...] = ("Section", "Question", "Answer")


def load_banking_data(file_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Read the banking FAQ dataset from a CSV file.

    Args:
        file_path: Path to the CSV.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing or the file is empty.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Empty answers would produce useless chunks that retrieval still returns.
    dataframe = dataframe.dropna(subset=["Section", "Question", "Answer"])
    dataframe = dataframe[
        dataframe["Answer"].astype(str).str.strip().ne("")
    ]

    if dataframe.empty:
        raise ValueError(f"Dataset contains no usable rows: {path}")

    return dataframe.reset_index(drop=True)


def clean_text(value: object) -> str:
    """Normalise whitespace in a cell value."""

    if value is None:
        return ""

    # Some rows contain stray newlines and repeated spaces from the source
    # data; collapsing them keeps the embedded text compact.
    return " ".join(str(value).split())


def create_documents(dataframe: pd.DataFrame) -> list[dict]:
    """Convert each CSV row into an embeddable document.

    Each row becomes one document because a Question/Answer pair is already a
    small, self-contained knowledge unit. Splitting it further would make
    retrieval worse.

    Returns:
        A list of {"text": str, "metadata": {"section": str}} dicts.
    """

    documents: list[dict] = []

    for row in dataframe.itertuples(index=False):
        section = clean_text(row.Section)
        question = clean_text(row.Question)
        answer = clean_text(row.Answer)

        if not question or not answer:
            continue

        documents.append(
            {
                "text": (
                    f"Section: {section}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                ),
                "metadata": {"section": section},
            }
        )

    logger.debug("Created %d documents.", len(documents))

    return documents


def main() -> int:
    """Print a short summary of the dataset. Returns an exit code."""

    dataframe = load_banking_data()
    documents = create_documents(dataframe)

    print(f"Total rows: {len(dataframe)}")
    print(f"Total documents: {len(documents)}")
    print(f"Sections: {sorted({d['metadata']['section'] for d in documents})}")

    print("\nFirst document:")
    print(documents[0]["text"])

    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())