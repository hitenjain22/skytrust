# What the experts say about sky events (and how SkyTrust uses it)

Notes from the organisations that publish sky events, gathered on 2026-09-30 for the Events and
Sky Guide pages. Quotes are verbatim from the pages listed under Sources. Anything SkyTrust
*computes* (times, rates, positions) is not repeated here: it comes from the code and is shown in
the app, with the checks in `tests/test_events.py`.

## International Meteor Organization (IMO): the meteor-shower calendar

The IMO's *2026 Meteor Shower Calendar* (J. Rendtel, ed.) is the reference list for meteor
showers; SkyTrust's `config/meteor_showers.yaml` is transcribed from its Table 5 ("Working List of
Visual Meteor Showers").

- **What a ZHR means:** "a calculated maximum number of meteors an ideal observer would see in
  perfectly clear skies (reference limiting magnitude +6.5) with the shower radiant overhead."
  So real rates are lower; SkyTrust shows the expected rate for each place with its light
  pollution and the Moon, using the IMO's correction formula.
- **The big three in 2026:** "the Quadrantid peak at Full Moon and moon-free nights for the peaks
  of the Perseids and the Geminids."
- **Autumn 2026:** "The maxima of the Orionids (008 ORI) and Leonids (013 LEO) occur when the Moon
  is near its first quarter. There is only a waxing crescent at the Geminid (004 GEM) maximum."
  The two Taurid branches peak "around November 05 (Southern Taurids…) and November 12 (Northern
  Taurids…), respectively, in a moon-free period."
- **Geminids:** "The best and most reliable of the major annual showers presently observable…
  known for bright meteors and fireballs. Well north of the equator, the radiant rises about
  sunset, reaching a usable elevation already from the local evening hours onwards."
- **Orionids:** "The shower's radiant… is at a useful elevation from local midnight or so…
  The waxing moon leaves some moon-free hours at least until the maximum of the shower."
- **Draconids:** "Perfectly moonless observing conditions… this year… Draconid meteors are
  exceptionally slow-moving." Usually weak, but storms in 1933 and 1946 and outbursts since.

## NASA

- **Meteor watching** (NASA's Orionids page): "Find an area well away from the city or street
  lights." "Lie flat on your back… and look up." "In less than 30 minutes in the dark, your eyes
  will adapt and you will begin to see meteors." The best views come from "45 to 90 degrees away
  from the radiant" where meteors "will appear longer and more spectacular." The show is viewable
  "during the hours after midnight" and "will last until dawn."
- **Shower parents:** the Orionids and the Eta Aquariids come from comet 1P/Halley; the
  Quadrantids "originate from an asteroid: asteroid 2003 EH1," and their peak "is much shorter –
  only a few hours." The Eta Aquariids are seen "during the pre-dawn hours."
- **October 2026 Night Sky Notes:** Algol, "the Demon Star," in Perseus, visible with Cassiopeia
  and Andromeda "in the northeastern sky, beginning after 9 PM," dims from magnitude +2.1 to +3.4
  "about every three days"; the Orionids are "active from Oct. 2 – Nov. 7, 2026, with peak night on
  Oct. 21-22."
- **Beginners:** NASA's tips pages suggest that "a good pair of binoculars" makes an excellent
  first instrument.
- **Eclipses:** NASA's eclipse catalogue (F. Espenak) gives the 2026 August 28 partial lunar
  eclipse an umbral magnitude of 0.9299, greatest at 04:13 UTC, which SkyTrust's computation
  reproduces.

## How SkyTrust turns this into advice

| Expert advice | In the app |
|---|---|
| Get away from city lights | Expected meteors per hour at every place, and a "best places in California" list for each shower |
| Moonlight hurts | Moon phase and whether it's up at the best time; a Krisciunas & Schaefer moonlight model in the sky chart |
| Look 45–90° from the radiant; lie back | Radiant direction and height for your place, and the tip on every shower |
| Give your eyes 20–30 minutes | In the beginner guide on Tonight |
| ZHR is an ideal-observer number | Rates shown for one observer, capped at the ZHR, never more |
| Peak nights are written as "Oct 21–22" | Events are dated by the evening the night starts |

## Sources

- IMO, *2026 Meteor Shower Calendar*: https://www.imo.net/the-2026-meteor-shower-calendar-is-here/
  (PDF copy: http://www.meteorastronomie.ch/images/cal2026_e.pdf)
- NASA, Orionids: https://science.nasa.gov/solar-system/meteors-meteorites/orionids/
- NASA, Quadrantids: https://science.nasa.gov/solar-system/meteors-meteorites/quadrantids/
- NASA, Eta Aquariids: https://science.nasa.gov/solar-system/meteors-meteorites/eta-aquarids/
- NASA Night Sky Network, October 2026: https://science.nasa.gov/solar-system/skywatching/night-sky-network/spooky-stargazing/
- NASA skywatching tips: https://science.nasa.gov/skywatching/tips-guides/
- NASA eclipses 2026: https://eclipse.gsfc.nasa.gov/OH/OH2026.html
