#!/usr/bin/env python3

"""
Download and save the Intelligence Squared (IQ2) debate corpus using ConvoKit.
After running this once, you'll have a local copy in the folder `iq2-corpus/`.
"""

from convokit import Corpus, download


def main():
    # This will automatically download the IQ2 corpus if not already cached.
    print("Downloading IQ2 corpus via ConvoKit (if not already cached)...")
    corpus = Corpus(download("iq2-corpus"))
    print("Download complete.")

    # Optional: print some basic info
    print("Number of conversations (debates):", len(corpus.conversations))
    print("Number of utterances (speeches/turns):", len(corpus.utterances))

    # Save corpus to a folder so you can reload it later without re-downloading
    out_dir = "/Volumes/TOSHIBA/MAD/data"
    print(f"Saving corpus to ./{out_dir} ...")
    corpus.dump(out_dir)
    print("Done. You can reload later with: Corpus('./iq2-corpus')")


if __name__ == "__main__":
    main()
