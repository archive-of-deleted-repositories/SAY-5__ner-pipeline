"""Seed the database with synthetic news-style Documents using Faker."""

from __future__ import annotations

import argparse
import random

from faker import Faker

from ner_pipeline.db import session_scope
from ner_pipeline.extractor import TransformersExtractor
from ner_pipeline.pipeline import ingest

TEMPLATES: tuple[str, ...] = (
    "{person} announced a new partnership between {org} and {org2} from offices in {city}, {country}.",
    "Authorities in {city} confirmed that {person} will lead {org}'s expansion across {country}.",
    "{org} reported quarterly results today; {person}, the chief executive, addressed analysts in {city}.",
    "Officials in {country} met with {person} to discuss {org}'s investment plans for {city}.",
    "Reporters in {city} caught up with {person} after the {org} board meeting concluded.",
)


def build_text(faker: Faker) -> str:
    template = random.choice(TEMPLATES)
    return template.format(
        person=faker.name(),
        org=faker.company(),
        org2=faker.company(),
        city=faker.city(),
        country=faker.country(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Insert documents without running the model (useful in CI / no-network).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    faker = Faker()
    Faker.seed(args.seed)

    if args.no_extract:
        from ner_pipeline.extractor import MockExtractor

        extractor = MockExtractor()
    else:
        extractor = TransformersExtractor()  # type: ignore[assignment]

    with session_scope() as session:
        for i in range(args.count):
            text = build_text(faker)
            res = ingest(
                source=f"seed-{i:04d}",
                text=text,
                extractor=extractor,
                session=session,
            )
            print(
                f"[{i+1:>4}/{args.count}] doc_id={res.document_id} "
                f"created={res.created} mentions={res.entity_count}"
            )


if __name__ == "__main__":
    main()
