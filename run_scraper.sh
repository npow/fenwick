#!/bin/bash

for team in ANA ATL-WPG BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA L.A MIN MTL N.J NSH NYI NYR OTT PHI PHX-ARI PIT S.J STL T.B TOR VAN WSH; do
  echo $team
  phantomjs scraper.js $team
done
wait
