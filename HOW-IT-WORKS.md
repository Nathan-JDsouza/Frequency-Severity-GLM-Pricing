# How this works (plain language)

This project asks a simple question about **car liability insurance** in France:

> If we look at who the driver is, what they drive, and where they live, can we tell **who is likely to cost the insurer more** — better than charging everyone the same?

The technical write-up is in [ANALYSIS.md](ANALYSIS.md). This page is for anyone who does not work in insurance math.

## The everyday idea

Imagine you run a garage that also sells a promise: “If this car hurts someone, we pay.” You cannot charge every customer the same if some groups crash much more often. You also cannot wait until after the crash to set the price. You need a **guess of the bill before it happens**.

That guess is not the sticker price on a website. It is only the **expected claim cost** no office expenses, no profit, no discounts. Actuaries call that a *pure premium*. Think of it as “what we think this policy will cost in claims.”

## Two questions, then multiply

A claim has two parts

1. **How often** does a policy have a claim?
2. **How large** is a typical claim?

Expected cost ≈ (expected number of claims) × (expected size of a claim).

We estimate those two pieces separately, then multiply. That is the whole model.

We use information already on the policy: area, region, how powerful and how old the car is, driver age, a **bonus-malus** score (a French no-claims / at-fault record — higher usually means a worse history), brand, fuel type, and how densely populated the area is.

## How we know it is not just a story we told ourselves

We hide **20%** of the policies, build the guess on the other **80%**, then score the hidden group. That is, we check if there's a pattern on 80% of the data, then use the remaining 20% to confirm the pattern.

We compare to a **flat rate**: take all the claim euros in that hidden group and spread them by how long each policy was in force. A car insured for a full year gets more of the pot than a car insured for a month. That is the “everyone the same per year” baseline.

Then we line policies up from “we think you will cost the most” to “we think you will cost the least” and look at **where the real claim euros actually went**.

## What we found

On that hidden 20% (about 136,000 policies):

- Real claims added up to about **€11.1 million**.
- The model’s expected costs added up to about **€12.3 million** (a bit high — we are better at **ranking** people than at hitting the exact euro total).
- The **10% of policies the model called most expensive** had about **1.5 times** as many claim euros per year of cover as the average. That is the main result: the ranking is not random.
- The ranking beats the flat rate. In everyday terms: **using the rating factors finds the costly policies better than ignoring them.**

Most of that signal is **how often** claims happen, not how large they are. People with a worse bonus-malus record have far more claims (about **0.08** per year of cover at the best band vs about **0.35** at the worst). Young drivers also have more claims. Claim *size* barely moves with those same factors.

## What this is not

- It is not “the price you would pay at a broker.” Real prices add expenses, tax, profit, and competition.
- It is not a guarantee. Insurance is luck plus pattern. A cheap-looking policy can still have one huge claim.
- The French files we use do not perfectly match “number of claims” on the policy to “list of claim amounts.” We did not paper over that.
- When we rank by **cost for this short policy term**, short-duration policies bunch at the bottom. A few large claims there make that group look wild. That is a ranking quirk, not “the cheapest 10% are six times worse.”

## One picture to remember

```
Who crashes more often?  ×  How big is a crash?  →  Expected cost
         (the strong part)        (the weak part)

Then: did the expensive-looking people actually cost more on data we hid?
Answer: yes, enough to beat “same price per year for everyone.”
```

## The stress test (September 2026)

We tried to break the model three ways. Plain-language results; numbers in [ANALYSIS.md](ANALYSIS.md#part-ii--model-validation-overdispersion-quasinb-corrections-and-tweedie).

### 1. “You’re counting claims wrong”

**Poisson** is the standard way to model *how many* claims happen. One assumption it makes: the spread around the average is exactly the average. We measured the actual spread (a “Pearson chi-square” test).

**Verdict: the assumption fails, and it matters — but only for confidence, not for prices.** The real spread is about **2.3×** what Poisson promises. And in a fun twist, the other common test (the “deviance ratio”) says the data looks *too quiet* — that test is simply broken on data this full of zeros, and we proved it with a simulation.

Where does the extra noise come from? Mostly from **1.35% of policies with phantom claims**: the count file says they had a claim, the money file says they didn’t. Those policies hold 27% of the claim *counts* but 0.2% of the claim *euros*. They aren't real risk — they're a paperwork mismatch between the two files.

We rebuilt the frequency model two ways that allow extra spread: **quasi-Poisson** (same prices, honest error bars) and **negative binomial** (slightly different prices). Result:

- Prices moved by less than 5% on any factor, and mostly much less.
- The “who is expensive” list barely changed: the top 10% of policies is **98.8% the same people**.
- But the model’s **stated confidence was fiction**: with the corrected error bars, a few factors (like neighbourhood density) that looked meaningful turn out to be noise.

### 2. “Price the whole thing in one model instead”

Instead of (how often?) × (how big?), a **Tweedie** model prices the total cost directly in one step. It has a dial, *p*, from 1 to 2, that controls how it balances “how likely any claim is” against “how big claims get.” We tried every setting 1.1 to 1.9 with cross-validation.

**Verdict: it ranks risks better, but prices the dollar level badly.**

- It ordered the hidden data better than the original model — better on 14 out of 15 test slices.
- But at the winning dial setting it predicted **€1.85 for every €1** of actual loss. A single correction factor (divide by 1.69) fixes the total, and after that it's slightly better calibrated overall.
- The two models disagree about **who** is expensive: only half of the top-10%-most-expensive policies are the same under both models. The Tweedie pushes younger, worse bonus-malus, longer-cover drivers higher.
- In the extreme tip — the most expensive 0.1% of policies — the original two-part model is clearly better: it finds **5.5%** of the hidden euros there vs the Tweedie's 2.5%.

### 3. “Are you just memorizing?”

We checked every model on data it never saw, repeatedly (15 different train/test slices).

**Verdict: no.** Every model scores a bit worse on unseen data (about 10–15% weaker), which is normal and small. Nothing here is memorizing.

### The bottom line

- **Keep the two-part model as the main price list.** The corrections didn't change it, and it's better in the extreme tail, which is where the money is.
- **Never trust the original model's confidence intervals** — use the corrected ones.
- **Fix the data before the model**: those phantom claims are the single biggest statistical problem, and no clever model fixes a broken join.
- **Keep the Tweedie as a challenger** to sanity-check the middle of the book.

```
   Frequency × Severity      Tweedie (one-step)
        │                        │
   best at the extremes     best through the middle
   honest euro totals       needs a level correction
   same prices after        different top-10% people
   correcting the noise     (only ~half overlap)
```

If you want the formulas, Gini numbers, and charts, read [ANALYSIS.md](ANALYSIS.md).
