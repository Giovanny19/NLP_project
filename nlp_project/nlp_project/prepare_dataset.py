"""
Step 1 & 2: Data Collection, Augmentation, and Labelling
- Loads original dataset (Kiswahili, English, Luo)
- Adds Sheng samples to meet project requirement
- Saves final labeled dataset as dataset_labeled.csv
"""

import pandas as pd
import os

# ─── Sheng samples (collected from Nairobi street language) ──────────────────
SHENG_SAMPLES = [
    "Niaje msee, mambo vipi?",
    "Sema buda, uko fit?",
    "Niko na ile chapaa ya kubeba.",
    "Leo nimepata dough mob, nashinda.",
    "Niko broke kabisa, hata fare haina.",
    "Nioshee kidogo, nitalipa wiki ijayo.",
    "Demu huyo ananipiga macho sana.",
    "Msee huyo ni rada sana, nakuhakikishia.",
    "Ile bash ilikuwa moto moto kabisa.",
    "Naskia unafanya side hustle gani siku hizi?",
    "Mimi najisort kila siku, bila mashaka.",
    "Kazi inanisumbua but lazima nihustle.",
    "Wacha tuende base ya msee mmoja.",
    "Usiku nacount coins, mchana nacheza game.",
    "Hii job si ya kudumu, natafuta ingine.",
    "Msee amenipea connect ya kazi fresh.",
    "Lazima upige kazi yako vizuri, usilegee.",
    "Siku hizi kila mtu anajaribu kuweza.",
    "Sijasort bado, lakini niko hapo hapo.",
    "Game inataka upige ngumi every day.",
    "Usiende njia ile usiku, ni hatari.",
    "Kwa base yetu hali ni ngumu lakini tunashinda.",
    "Matatu zinaenda speed mob usiku.",
    "Polisi walikuja jana usiku, kelele mob.",
    "Angalia, msee huyo si wa kuamini.",
    "Usimwambie siri, ataenda na story yote.",
    "Hii deal inaonekana na harufu mbaya.",
    "Usimwamini, ana kauli mbili kabisa.",
    "Wacha umbeya, utajisumbua tu bure.",
    "Msee huyo ana story nyingi sana, jiepushe.",
    "Naomba nikule ugali na samaki wa leo.",
    "Njaa inaniua, tukule kitu haraka haraka.",
    "Hii ngoma imeniingia deep kabisa, fire.",
    "Outfit yake ilikuwa fire leo usiku sana.",
    "Sisi tulikuwa tukienjoy mpaka alfajiri.",
    "Hii beat inanisort, nakuambia ukweli.",
    "Tulipiga mdundo mpaka miguu iliuma.",
    "Hii vibe ni tamu, usiiache iende.",
    "Macho wazi kwa hii mtaa, usijisahau.",
    "Niaje demu, uko poa leo?",
    "Wee rafiki, unanikimbia au?",
    "Sema tu ukweli, tunapiga sawa?",
    "Ongea nawe baadaye, sawa msee.",
    "Nitakupiga mzigo hii usiku, ngoja.",
    "Niko na mkopo wa kufunga hii wiki.",
    "Pesa imekwisha, nje ya count kabisa.",
    "Usiniambie bei ngumu, tufanye discount.",
    "Ile business yake inabeba vizuri sana.",
    "Huyo msee amenifall deep sana kweli.",
    "Tukiongea usiku wote, time inapita haraka.",
    "Amenivunjia moyo mara nyingi sana buda.",
    "Huyo demu ni type yangu kabisa.",
    "Nikimuona roho yangu inacheza haraka.",
    "Kwa hapa ukijua mtu mmoja, unajua wote.",
    "Vijana wa kona wanajuana wote sawa.",
    "Biashara ya mtaani inahitaji subira nyingi.",
    "Jua kali masters wanafanya kazi ya ajabu.",
    "Soda ya baridi baada ya kazi ni salama.",
    "Hata mkate na chai inaniridhisha usiku.",
    "Fry ya viazi na nyanya ni bora kuliko steak.",
    "Kujiambia ukweli, hustle yangu inaenda poa.",
    "Nitaingia ofisini kesho, nina meeting ya moto.",
    "Wee, nishike form ya haraka sana.",
    "Uko rada, chief wa mtaa?",
    "Haiya, umerudi lini kutoka?",
    "Mambo vipi msee wa kando yangu?",
    "Nipigie ikiwa uko free, sawa buda.",
    "Tutaonana kwa round ya jioni leo.",
    "Naskia mama amepika pilau leo jioni.",
    "Ukijaribu kushoto na kulia, utaweza kabisa.",
    "Hii life ni mzuri ikiwa unajua kucheza.",
    "Watu wa kando wanajua kila kitu kabisa.",
    "Usitembee na macho chini kwa hapa.",
    "Hii kitu inanigharimu mbaya sana.",
    "Doh yangu iko kwa savings, sishike cash.",
    "Naskia umepata deal ya mafuta, congrats!",
    "Tukisimama pamoja tunaonekana moto kabisa.",
    "Nilipiga selfie na madem wazuri sana.",
    "Msee huyo ana swag ya pekee kabisa.",
    "Siku ya leo imeniletea mood ya top.",
    "Tulifanya date ya sawa saa ile usiku.",
    "Mpenzi wangu yu mbali lakini nampenda.",
    "Sisi wawili tuko sawa, hakuna story yoyote.",
    "Ukimwambia, kesho asubuhi kila mtu atajua.",
    "Hii business ina wasiwasi, jiangalie sana.",
    "Usifanye haraka, fikiria kwanza kabla ya kitu.",
    "Base yangu iko mbali na CBD kidogo.",
    "Hii mtaa ina watoto wa mtaani wengi.",
    "Jiangalie ukiniambia nini kinaendelea.",
    "Leo tumeenda party ya moto, twende.",
    "Msee wangu alifanya kitu kibaya sana.",
    "Sawa sawa, tutaonana baadaye kwa base.",
    "Niko na mtu, nitakupigia baadaye buda.",
    "Enda polepole, game bado iko mapema.",
    "Wacha niangalie kitu kidogo, ngoja.",
    "Nimechoka sana, nataka kulala sasa hivi.",
    "Wewe ni mtu wa ajabu, nakuambia.",
    "Kila kitu ni sawa, usijali kabisa.",
    "Hii ni story ndefu, tutaongea baadaye.",
    "Sina muda wa kucheza mchezo hii.",
    "Niko busy sana leo, tuongee kesho buda.",
]

def prepare_dataset():
    print("=" * 55)
    print("  STEP 1 & 2: DATA COLLECTION & LABELLING")
    print("=" * 55)

    # Load original dataset
    df_original = pd.read_csv(
        "Dataset.csv",
        header=None,
        names=["language", "text"]
    )

    # Drop NaN rows
    df_original.dropna(subset=["language", "text"], inplace=True)
    df_original.reset_index(drop=True, inplace=True)

    print(f"\nOriginal dataset rows : {len(df_original)}")
    print("Original distribution :")
    print(df_original["language"].value_counts().to_string())

    # Build Sheng dataframe
    df_sheng = pd.DataFrame({
        "language": ["Sheng"] * len(SHENG_SAMPLES),
        "text":     SHENG_SAMPLES
    })

    # Combine (keep duplicates to preserve class balance)
    df = pd.concat([df_original, df_sheng], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    # Ensure plain Python string dtype (avoid Arrow StringArray issues)
    df["language"] = df["language"].astype(str)
    df["text"]     = df["text"].astype(str)

    print(f"\nFinal dataset rows    : {len(df)}")
    print("Final distribution    :")
    print(df["language"].value_counts().to_string())

    # Save
    df.to_csv("dataset_labeled.csv", index=False)
    print(f"\n✓ Saved → dataset_labeled.csv")
    return df


if __name__ == "__main__":
    prepare_dataset()
