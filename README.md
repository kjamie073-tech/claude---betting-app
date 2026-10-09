# Premier League bet builder model

A Premier League match model for building and checking bet builders: it
estimates the probability of every common market (result, goals, BTTS,
corners, cards, shots, player goals, assists and bookings), prices
combinations of legs together so linked picks are handled properly, and
compares everything with the bookmaker's odds.

Data is downloaded daily by the `Update data` GitHub Action and stored on the
`data` branch.
