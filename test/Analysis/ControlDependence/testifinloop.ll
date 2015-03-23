; RUN: opt < %s -print-controldep -analyze | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
define void @testifinloop() {
start:
  br label %loop
loop:
  br label %loopbody
loopbody:
  br label %loopif
loopif:
  br i1 true, label %looptrue, label %loopfalse
looptrue:
  br label %loopifmerge
loopfalse:
  br label %loopifmerge
loopifmerge:
  br label %looptest
looptest:
  br i1 true, label %loop, label %returnit
returnit:
  ret void
}
; CHECK: ({loopfalse},{looptrue},{returnit},{start},{loop,loopbody,loopif,loopifmerge,looptest})
