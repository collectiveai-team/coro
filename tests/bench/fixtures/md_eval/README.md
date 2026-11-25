# Offline MeetEval scorer

`md-eval-22.pl` comes from [the dscore revision previously used in CI](https://github.com/nryant/dscore/blob/3207c6c65a9b48429dcb52e720edcabfeb3d3f35/scorelib/md-eval-22.pl).
The fixture normalizes line endings and whitespace. The upstream license is in
`LICENSE`.

Bench tests prepend this directory to `PATH` so MeetEval finds the executable
before it tries to download the script. MeetEval runs it with Perl, which must
be installed locally.
