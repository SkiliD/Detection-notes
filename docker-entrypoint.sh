#!/bin/sh
# Entrypoint: MuseScore (PDF engraving, --pdf) needs a display even headless,
# so wrap in xvfb-run only when PDF output is requested. Do NOT exec the
# wrapper: xvfb-run as container PID 1 hangs in its Xvfb cleanup; keep this
# shell as PID 1 and run xvfb-run as a plain child.
case " $* " in
  *" --pdf "*) xvfb-run -a v2m "$@" ;;
  *) exec v2m "$@" ;;
esac
