#!/bin/sh
set -e

# コンテナ起動時に渡されたコマンドがあれば、それを実行する。
# なければ、コンテナが終了しないようにsleepし続ける。
if [ "$#" -gt 0 ]; then
    exec python main.py "$@"
else
    exec sleep infinity
fi
