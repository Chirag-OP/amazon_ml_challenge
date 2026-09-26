import re
import pandas as pd
from datasketch import MinHash, MinHashLSH


class EntityLSHIndex:

    def __init__(
        self,
        num_perm=64,
        shingle_size=3,
        threshold=0.5
    ):
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.threshold = threshold

        # Structure:
        #
        # self.indexes[country]["name"]
        # self.indexes[country]["address"]
        #
        self.indexes = {}

        # Store MinHash objects by:
        #
        # self.minhashes["name"][entity_id]
        # self.minhashes["address"][entity_id]
        #
        self.minhashes = {
            "name": {},
            "address": {}
        }

    def normalize(self, text):

        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def get_minhash(self, text):

        text = self.normalize(text)

        m = MinHash(num_perm=self.num_perm)

        if not text:
            return m

        # Character n-grams
        for i in range(
            len(text) - self.shingle_size + 1
        ):

            shingle = text[
                i:i + self.shingle_size
            ]

            m.update(
                shingle.encode("utf8")
            )

        return m

    def build(
        self,
        df,
        id_col="entity_id",
        country_col="country",
        name_col="business_name",
        address_col="business_address"
    ):

        countries = (
            df[country_col]
            .dropna()
            .unique()
        )

        print("=" * 60)
        print("Countries found:")
        print(list(countries))
        print("=" * 60)

        for country in countries:

            country_df = df[
                df[country_col] == country
            ]

            print(
                f"\nBuilding LSH indexes for {country}"
            )

            print(
                f"Number of records: {len(country_df)}"
            )

            # Create two LSH indexes
            #
            # country
            #   ├── name
            #   └── address
            #

            self.indexes[country] = {

                "name": MinHashLSH(
                    threshold=self.threshold,
                    num_perm=self.num_perm
                ),

                "address": MinHashLSH(
                    threshold=self.threshold,
                    num_perm=self.num_perm
                )
            }

            for _, row in country_df.iterrows():

                entity_id = str(
                    row[id_col]
                )

                name = row[name_col]

                if pd.notna(name):

                    name_minhash = (
                        self.get_minhash(name)
                    )

                    # Store actual entity ID
                    self.indexes[country][
                        "name"
                    ].insert(
                        entity_id,
                        name_minhash
                    )

                    # Store MinHash for later similarity
                    self.minhashes[
                        "name"
                    ][entity_id] = name_minhash


                address = row[address_col]

                if pd.notna(address):

                    address_minhash = (
                        self.get_minhash(address)
                    )

                    # Store actual entity ID
                    self.indexes[country][
                        "address"
                    ].insert(
                        entity_id,
                        address_minhash
                    )

                    # Store MinHash for later similarity
                    self.minhashes[
                        "address"
                    ][entity_id] = address_minhash

            print(
                f"Finished {country}"
            )

        print("\n" + "=" * 60)
        print("LSH indexes built successfully")
        print("=" * 60)

    def query(
        self,
        country,
        business_name,
        business_address
    ):


        if country not in self.indexes:

            return {
                "name": set(),
                "address": set(),
                "union": set()
            }

        name_minhash = self.get_minhash(
            business_name
        )

        name_candidates = set(
            self.indexes[country]["name"].query(
                name_minhash
            )
        )

        address_minhash = self.get_minhash(
            business_address
        )

        address_candidates = set(
            self.indexes[country]["address"].query(
                address_minhash
            )
        )

        all_candidates = (
            name_candidates |
            address_candidates
        )

        return {
            "name": name_candidates,
            "address": address_candidates,
            "union": all_candidates
        }

    def get_similarity(
        self,
        field,
        entity_id_1,
        entity_id_2
    ):
        """
        Jaccard similarity between two entities that are BOTH already
        indexed (e.g. comparing two Dataset-2 records to each other).
        """

        mh1 = self.minhashes[
            field
        ].get(entity_id_1)

        mh2 = self.minhashes[
            field
        ].get(entity_id_2)

        if mh1 is None or mh2 is None:
            return None

        return mh1.jaccard(mh2)

    def similarity_to_candidate(
        self,
        field,
        query_minhash,
        candidate_id
    ):
        """
        Jaccard similarity between a QUERY record's MinHash (typically a
        Dataset-1 record that was never inserted into this index) and an
        already-indexed candidate's stored MinHash.

        NOTE: pass in a MinHash you've already computed via get_minhash()
        rather than raw text -- computing it fresh on every candidate
        inside a ranking loop means recomputing the same shingle hash
        over and over for no reason.
        """

        candidate_minhash = self.minhashes[field].get(candidate_id)

        if candidate_minhash is None:
            return 0.0

        return query_minhash.jaccard(candidate_minhash)