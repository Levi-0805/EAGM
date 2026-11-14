# GitHub transport diagnosis, 2026-11-14

Browser sign-in and command-line network reachability were separate issues.
The existing local hosts interception affected GitHub hostnames, and command-line
TLS did not trust the intercepted certificate. A direct public-IP route previously
worked but later failed with TLS termination/timeouts. Merely logging in again
did not change either route.

The currently configured local HTTP proxy at 127.0.0.1:7897 is reachable. Sending
it a hostname reintroduced the interception; sending it a public GitHub IP while
retaining github.com SNI and certificate verification returned HTTP 200.

The GitHub-only temporary CONNECT helper was updated to resolve a public IP and
CONNECT that IP through the configured local proxy. Client TLS stays end-to-end;
no sslVerify=false, untrusted certificate installation, system hosts edit or
credential export is used. A backup of the previous helper is retained at
`/home/aaa/.local/share/github-direct/run.py.before-proxy-chain`.

The first development commit `5037c4904dcd3ee99d15ab95e31c15cac413de94` was
successfully pushed to `Levi-0805/EAGM`, branch `EAGM-2.0-full-validation`.
Failed earlier push attempts did not lose the local commit or stop experiments.
