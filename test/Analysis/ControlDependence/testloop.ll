; RUN: opt < %s -print-controldep -analyze | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"

define void @testbasicloop() {
start:
  br label %loop
loop:
  br label %loopbody
loopbody:
  br label %looptest
looptest:
  br i1 true, label %loop, label %returnit
returnit:
  ret void
}
; CHECK: ({returnit},{start},{loop,loopbody,looptest})
