; RUN: opt -basicaa -memoryssa -dump-memoryssa -verify-memoryssa -disable-output < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
; Function Attrs: ssp uwtable
define i32 @main() #0 {
entry:
; CHECK:  2 = MemoryDef(1)
; 2 = MemoryDef(1)
  %call = call noalias i8* @_Znwm(i64 4) #2
  %0 = bitcast i8* %call to i32*
; CHECK:  3 = MemoryDef(2)
; 3 = MemoryDef(2)
  %call1 = call noalias i8* @_Znwm(i64 4) #2
  %1 = bitcast i8* %call1 to i32*
; These stores should conflict
; CHECK:  4 = MemoryDef(3)
; 4 = MemoryDef(3)
  store i32 5, i32* %0, align 4
; CHECK:  5 = MemoryDef(4)
; 5 = MemoryDef(4)
  store i32 7, i32* %1, align 4
; Because we optimize uses, these uses should not be of the last store, but of
; the conflicting store for each one
; CHECK:  MemoryUse(4)
; MemoryUse(4)
  %2 = load i32, i32* %0, align 4
; CHECK:  MemoryUse(5)
; MemoryUse(5)
  %3 = load i32, i32* %1, align 4
; CHECK:  MemoryUse(4)
; MemoryUse(4)
  %4 = load i32, i32* %0, align 4
; CHECK:  MemoryUse(5)
; MemoryUse(5)
  %5 = load i32, i32* %1, align 4
  %add = add nsw i32 %3, %5
  ret i32 %add
}
declare noalias i8* @_Znwm(i64) #1

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nobuiltin "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { builtin }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (http://llvm.org/git/clang.git c9903b44534b8b5debcdbe375ee5c1fec6cc7243) (llvm/trunk 228022)"}
