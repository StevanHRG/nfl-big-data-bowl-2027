# Bend You Can Measure: A Combine Turn-Force Score That Predicts NFL Pressure

## 1. The trait scouts can see but can't time

Ask any defensive line coach what separates a pass rusher who wins from one who doesn't, and you'll hear one word: **bend**. It means turning the corner around an offensive tackle at full speed without drifting wide or slowing down. Scouts grade bend by eye. The stopwatch can't see it. The 40-yard dash and 10-yard split measure straight lines, and the 3-cone drill mixes turning, stopping and starting into one number (and 65% of prospects in this data skipped it).

The 2027 Big Data Bowl gives us something new. The Combine's position drills were tracked at 10 Hz. Several of the defensive line drills are built around exactly this skill: running an arc at speed. That lets us turn "bend" into physics.

**Our question:** does the force a defensive lineman can generate while turning in Combine drills predict how often he pressures NFL quarterbacks, **beyond** what his draft slot, weight, alignment and straight-line speed already tell us?

**Our answer:** yes. We introduce the **Combine Bend Score (CBS)**. One standard deviation of CBS is worth about **+1 pressure per 100 pass rushes**, roughly 13% of the baseline pressure rate for these young rushers. The effect is independent of draft slot and the 10-yard split. It survives a validation in which the drills are chosen without seeing the test class. And it flags late-round rushers who outperform their draft slot.

## 2. Measuring bend: centripetal acceleration

When a body follows a curve, the force pulling it toward the center of the turn is **centripetal acceleration**:

> a_c = v · ω  (speed × turning rate) = v² / r

A rusher who keeps his speed (high *v*) through a tight turn (high *ω*, small *r*) produces a large a_c. That is bend, in yards per second squared. We compute it at every tracking frame:

- heading comes from the x/y path;
- turning rate is the frame-to-frame change in heading;
- the result is smoothed over 3 frames (0.3 s), because a single 0.1 s frame is dominated by sensor noise.

For each player and drill, we keep the **peak** a_c across attempts.

Three DL drills are arc drills by design, and we use those:

| Drill | Movement in the tracking data | Mean peak a_c |
|---|---|---|
| **Run the Hoop** | a sprinted figure-eight | 7.3 yd/s² |
| **Run and Club** | a sprinted arc with a sharp redirection | 6.2 yd/s² |
| **Front Start Wave + Lateral Reaction** | a burst from a stance followed by lateral changes of direction | 6.2 yd/s² |

The **Combine Bend Score** is the average of the player's z-scores across these drills. It is rescaled so that 50 is average and each 10 points is one standard deviation. Because it comes from position drills that nearly every DL performs, CBS exists for **114 of 118** DL prospects, versus 90 with a timed 10-yard split.

![Figure 1](fig1_hoop_paths.gif)

*Figure 1. Two Run the Hoop attempts in real time. Both players trace nearly the same path, but the sensor shows the difference: Cedric Johnson (pick #214) peaks at 7.4 yd/s², Brenton Cox (undrafted) at 6.4. Each dot is 0.1 s; darker blue means more turning force.*

## 3. Does bend matter on Sundays?

Before linking the Combine to the NFL, we check that the trait matters in games. We measured the same quantity, peak a_c in the first 3 seconds after the snap, on **33,249 regular-season pass rushes** by DL from the 2023–25 classes.

![Figure 2](fig2_ingame_mechanism.png)

*Figure 2. Pressure rate rises sharply once a rusher's in-game turn force passes about 5 yd/s². This holds for edge and interior rushers alike.*

At the play level, each extra yd/s² of turn force raises the odds of a pressure by about **15%** (logistic regression with alignment controls, p < 10⁻²⁰). This is a *mechanism* check, not a causal claim. A rusher who has already beaten his blocker will naturally turn toward the quarterback. But it confirms that the movement we measure at the Combine is the movement that produces pressure.

## 4. From Indianapolis to Sundays

**Outcome.** For each DL with at least 50 regular-season pass rushes (79 players, 31,111 rushes, 2,367 pressures), we compute **pressures over expected (POE) per 100 rushes**. "Expected" is the average rate of all DL in our sample for his alignment on each snap, so edges are compared with edges and interior linemen with interior linemen. To avoid rewarding small samples, each player's rate is shrunk toward zero with 100 pseudo-rushes.

**Result.** Players in the top quartile of CBS pressured **3.6 more times per 100 rushes** than the bottom quartile (95% bootstrap CI 1.9 to 4.9).

![Figure 3](fig3_quartiles.png)

*Figure 3. Pressure over expected rises steadily with Combine Bend Score.*

**Does CBS just repackage information teams already have?** We regressed POE on CBS together with:

- draft slot (log of overall pick, which captures how highly teams valued the player);
- weight;
- share of rushes taken from the edge;
- the 10-yard split.

Each player was weighted by his number of rushes.

![Figure 4](fig4_coefficients.png)

*Figure 4. Effect of +1 SD of each trait, all estimated together (n = 63 DL with every measure available).*

CBS (**+0.99**, 95% CI 0.45 to 1.53) and the 10-yard split (+1.08) both carry independent signal. The draft slot adds little once these are known. CBS is only weakly related to draft position (r = 0.16), which suggests **teams are not fully pricing bend**. Its correlation with the 10-yard split is moderate (r = 0.42), so they are different traits: one is getting off the ball, the other is turning the corner.

## 5. Is it real? Stress-testing a small sample

With about 80 players and many possible drills, the main risk is a lucky pattern. Here is everything we tested, including what failed.

**Nested validation (Figure 5).** We hid one draft class at a time. Using only the other two classes, an automated rule selected which DL drills showed signal. We then built the score and predicted pressure for the unseen class. This repeats the drill selection inside every fold, so the test class never influences which drills are used. The procedure reduced out-of-sample prediction error by **5.7%** compared with a model using only draft slot, weight and alignment. In 1,000 permutations, where turn force was shuffled across players and the whole procedure rerun, only **1.5%** did as well (p = 0.015).

![Figure 5](fig5_validation.png)

*Figure 5. Nested, out-of-class validation against 1,000 permutations.*

**Robustness:**

- **Not just top speed.** Controlling for each player's maximum speed in the same drills barely moves the CBS coefficient (0.41 → 0.40 SD, p < 0.001).
- **Smoothing.** The result holds with 3- and 5-frame smoothing (p < 0.001). With no smoothing (single frames) it weakens to p = 0.10, as expected from frame-level noise.
- **Peak, not typical.** Building CBS from each attempt's 90th-percentile turning force instead of its peak weakens the signal (p = 0.13). What predicts pressure is the hardest turn a player can produce, not his average curve.
- **Arbitrary choices.** Across 27 combinations of shrinkage strength (50, 100 or 200 pseudo-rushes), minimum sample (30, 50 or 100 rushes) and the pick imputed for undrafted players, the CBS effect stays between +0.83 and +1.11 POE/100, and every 95% interval excludes zero.
- **Every class.** With the three arc drills fixed, the CBS coefficient is positive in every leave-one-class-out fit and improves prediction of each held-out class. Those gains (15–19%) are larger than the nested 5.7% because the drill choice saw all classes; the nested result is the one to trust.

**What did not work.** We report these because they shape what CBS means:

- **Pure burst.** "Time to accelerate from 1 to 4 yd/s" in DL drills added nothing once weight and alignment were controlled. Explosiveness without the turn is not the signal.
- **Non-arc drills.** The Four Bag, Pass Rush and Body Control drills, taken alone, point the same direction but weakly (combined p = 0.25). **The signal lives in drills that require turning at speed.**
- **Other positions.** The same method on offensive linemen (pressure allowed) found nothing (p = 0.17). An analogous "brake-and-burst" score for receivers depended on a single route and did not survive removing it.

## 6. Putting it to work

### On draft weekend: finding the next Day-3 rusher

![Figure 6](fig6_scouting_board.png)

*Figure 6. Every 2023–25 DL prospect, by draft slot and Combine Bend Score. The top-right quadrant marks late picks with elite bend.*

Among rushers drafted after pick 100 or undrafted, those with **CBS ≥ 60** pressured **right at the average for their alignment (+0.03 POE/100)**. The other late picks fell **2.1 pressures per 100 rushes short**. Examples:

- **Cedric Johnson** (#214, CBS 68): +2.9 POE/100.
- **Javon Solomon** (#168, CBS 67): +2.2.
- **Adetomiwa Adebawore** (#110, CBS 66): +1.5 over 511 rushes.

CBS is not a guarantee. **Robert Beal** (#173, CBS 64) underperformed. Bend is one ingredient, alongside hand usage, power and processing. But it is an ingredient a stopwatch cannot see, and it is measured on drills almost every prospect already runs.

**How an evaluator uses it:** CBS acts as a tiebreaker and a re-check flag. When two Day-3 rushers grade similarly on film, take the one with higher CBS. When a prospect with CBS ≥ 60 sits low on the board, send the film back for another look at his corner work.

### Week to week: the same metric as a development KPI

The calculation that produces CBS also runs on in-game tracking, and at the play level in-game turn force is strongly tied to pressure (Section 3). That suggests a weekly use: chart each rusher's **peak turn force per rush** next to his pressures, **compared against rushers in the same alignment**. Pressures are noisy and arrive slowly; turn force is measured on every rush. One caution from our data: a raw season average of in-game turn force is dominated by size and alignment, and did not predict pressure on its own. Any weekly KPI has to be read relative to role, which is exactly how we built CBS. For a young rusher with a low CBS, the arc drills that measure it double as a **development protocol**: a repeatable test of whether his bend is improving, read in the same units as the Combine.

## 7. Limitations

- **Sample size.** 79 DL with enough snaps. Effects are moderate and the intervals are wide. We prioritized out-of-class validation over in-sample p-values for that reason.
- **Selection.** Only players who made NFL rosters and earned pass-rush snaps have outcomes. Low-CBS players who never played are invisible, which likely *understates* CBS's value.
- **Snap timing.** Combine drill clips sometimes begin mid-motion. That is why we use peak turning force rather than time-from-start measures.
- **Outcome.** Pressure depends on scheme, teammates and opponents. Alignment adjustment and shrinkage reduce this noise but cannot remove it.

## 8. Conclusion

Bend has always been the trait scouts trust but cannot time. With Combine tracking, it becomes a number: **centripetal acceleration in the arc drills**.

- It predicts NFL pass-rush pressure beyond draft slot, size and straight-line speed.
- It holds up under nested, out-of-class validation.
- It is available for nearly every DL prospect.
- It points to late-round rushers the market undervalued.

The same calculation runs on game tracking, so it can follow the player from Indianapolis to his Sundays.

---

**Appendix & code:** all analysis is reproducible in the attached public Kaggle notebook. This includes feature extraction, the nested permutation test, robustness checks and figures.
