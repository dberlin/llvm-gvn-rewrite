; RUN: opt < %s -print-controldep -analyze | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
define void @teststraightline() {
start:
    br label %next
next:
    br label %returnit
returnit:
    ret void
}
; CHECK: ({next,returnit,start})

