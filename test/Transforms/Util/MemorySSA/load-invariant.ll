; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
;
; Invariant loads should be considered live on entry, because, once the
; location is known to be dereferenceable, the value can never change.

@g = external global i32

define i32 @foo() {
  %1 = alloca i32, align 4
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %1, align 4
  %2 = alloca i32, align 4

; CHECK: MemoryUse(1)
; CHECK-NEXT: %3 = load i32
  %3 = load i32, i32* %1, align 4
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %4 = load i32
  %4 = load i32, i32* %2, align 4, !invariant.load !0
  %5 = add i32 %3, %4
  ret i32 %5
}

!0 = !{}
