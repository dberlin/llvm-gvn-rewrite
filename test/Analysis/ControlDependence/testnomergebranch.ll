; RUN: opt < %s -print-controldep -analyze | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
define void @testnomergebranch() {
start:
    br i1 true, label %same, label %different
same:
    ret void
different:
    ret void
}
; CHECK: ({different},{same},{start})

