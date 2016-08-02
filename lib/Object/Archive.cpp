//===- Archive.cpp - ar File Format implementation --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ArchiveObjectFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Archive.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace object;
using namespace llvm::support::endian;

static const char *const Magic = "!<arch>\n";
static const char *const ThinMagic = "!<thin>\n";

void Archive::anchor() { }

static Error
malformedError(Twine Msg) {
  std::string StringMsg = "truncated or malformed archive (" + Msg.str() + ")";
  return make_error<GenericBinaryError>(std::move(StringMsg),
                                        object_error::parse_failed);
}

ArchiveMemberHeader::ArchiveMemberHeader(const Archive *Parent,
                                         const char *RawHeaderPtr,
                                         uint64_t Size, Error *Err)
    : Parent(Parent),
      ArMemHdr(reinterpret_cast<const ArMemHdrType *>(RawHeaderPtr)) {
  if (RawHeaderPtr == nullptr)
    return;
  ErrorAsOutParameter ErrAsOutParam(Err);

  if (Size < sizeof(ArMemHdrType)) {
    if (Err) {
      std::string Msg("remaining size of archive too small for next archive "
                      "member header ");
      Expected<StringRef> NameOrErr = getName(Size);
      if (!NameOrErr) {
        consumeError(NameOrErr.takeError());
        uint64_t Offset = RawHeaderPtr - Parent->getData().data();
        *Err = malformedError(Msg + "at offset " + Twine(Offset));
      } else
        *Err = malformedError(Msg + "for " + NameOrErr.get());
    }
    return;
  }
  if (ArMemHdr->Terminator[0] != '`' || ArMemHdr->Terminator[1] != '\n') {
    if (Err) {
      std::string Buf;
      raw_string_ostream OS(Buf);
      OS.write_escaped(llvm::StringRef(ArMemHdr->Terminator,
                                       sizeof(ArMemHdr->Terminator)));
      OS.flush();
      std::string Msg("terminator characters in archive member \"" + Buf +
                      "\" not the correct \"`\\n\" values for the archive "
                      "member header ");
      Expected<StringRef> NameOrErr = getName(Size);
      if (!NameOrErr) {
        consumeError(NameOrErr.takeError());
        uint64_t Offset = RawHeaderPtr - Parent->getData().data();
        *Err = malformedError(Msg + "at offset " + Twine(Offset));
      } else
        *Err = malformedError(Msg + "for " + NameOrErr.get());
    }
    return;
  }
}

// This gets the raw name from the ArMemHdr->Name field and checks that it is
// valid for the kind of archive.  If it is not valid it returns an Error.
Expected<StringRef> ArchiveMemberHeader::getRawName() const {
  char EndCond;
  auto Kind = Parent->kind();
  if (Kind == Archive::K_BSD || Kind == Archive::K_DARWIN64) {
    if (ArMemHdr->Name[0] == ' ') {
      uint64_t Offset = reinterpret_cast<const char *>(ArMemHdr) -
                        Parent->getData().data();
      return malformedError("name contains a leading space for archive member "
                            "header at offset " + Twine(Offset));
    }
    EndCond = ' ';
  }
  else if (ArMemHdr->Name[0] == '/' || ArMemHdr->Name[0] == '#')
    EndCond = ' ';
  else
    EndCond = '/';
  llvm::StringRef::size_type end =
      llvm::StringRef(ArMemHdr->Name, sizeof(ArMemHdr->Name)).find(EndCond);
  if (end == llvm::StringRef::npos)
    end = sizeof(ArMemHdr->Name);
  assert(end <= sizeof(ArMemHdr->Name) && end > 0);
  // Don't include the EndCond if there is one.
  return llvm::StringRef(ArMemHdr->Name, end);
}

// This gets the name looking up long names. Size is the size of the archive
// member including the header, so the size of any name following the header
// is checked to make sure it does not overflow.
Expected<StringRef> ArchiveMemberHeader::getName(uint64_t Size) const {

  // This can be called from the ArchiveMemberHeader constructor when the
  // archive header is truncated to produce an error message with the name.
  // Make sure the name field is not truncated.
  if (Size < offsetof(ArMemHdrType, Name) + sizeof(ArMemHdr->Name)) {
    uint64_t ArchiveOffset = reinterpret_cast<const char *>(ArMemHdr) -
                      Parent->getData().data();
    return malformedError("archive header truncated before the name field "
                          "for archive member header at offset " +
                          Twine(ArchiveOffset));
  }

  // The raw name itself can be invalid.
  Expected<StringRef> NameOrErr = getRawName();
  if (!NameOrErr)
    return NameOrErr.takeError();
  StringRef Name = NameOrErr.get();

  // Check if it's a special name.
  if (Name[0] == '/') {
    if (Name.size() == 1) // Linker member.
      return Name;
    if (Name.size() == 2 && Name[1] == '/') // String table.
      return Name;
    // It's a long name.
    // Get the string table offset.
    std::size_t StringOffset;
    if (Name.substr(1).rtrim(' ').getAsInteger(10, StringOffset)) {
      std::string Buf;
      raw_string_ostream OS(Buf);
      OS.write_escaped(Name.substr(1).rtrim(' '));
      OS.flush();
      uint64_t ArchiveOffset = reinterpret_cast<const char *>(ArMemHdr) -
                               Parent->getData().data();
      return malformedError("long name offset characters after the '/' are "
                            "not all decimal numbers: '" + Buf + "' for "
                            "archive member header at offset " +
                            Twine(ArchiveOffset));
    }

    // Verify it.
    if (StringOffset >= Parent->getStringTable().size()) {
      uint64_t ArchiveOffset = reinterpret_cast<const char *>(ArMemHdr) -
                               Parent->getData().data();
      return malformedError("long name offset " + Twine(StringOffset) + " past "
                            "the end of the string table for archive member "
                            "header at offset " + Twine(ArchiveOffset));
    }
    const char *addr = Parent->getStringTable().begin() + StringOffset;

    // GNU long file names end with a "/\n".
    if (Parent->kind() == Archive::K_GNU ||
        Parent->kind() == Archive::K_MIPS64) {
      StringRef::size_type End = StringRef(addr).find('\n');
      return StringRef(addr, End - 1);
    }
    return addr;
  }

  if (Name.startswith("#1/")) {
    uint64_t NameLength;
    if (Name.substr(3).rtrim(' ').getAsInteger(10, NameLength)) {
      std::string Buf;
      raw_string_ostream OS(Buf);
      OS.write_escaped(Name.substr(3).rtrim(' '));
      OS.flush();
      uint64_t ArchiveOffset = reinterpret_cast<const char *>(ArMemHdr) -
                        Parent->getData().data();
      return malformedError("long name length characters after the #1/ are "
                            "not all decimal numbers: '" + Buf + "' for "
                            "archive member header at offset " +
                            Twine(ArchiveOffset));
    }
    if (getSizeOf() + NameLength > Size) {
      uint64_t ArchiveOffset = reinterpret_cast<const char *>(ArMemHdr) -
                        Parent->getData().data();
      return malformedError("long name length: " + Twine(NameLength) +
                            " extends past the end of the member or archive "
                            "for archive member header at offset " +
                            Twine(ArchiveOffset));
    }
    return StringRef(reinterpret_cast<const char *>(ArMemHdr) + getSizeOf(),
                     NameLength).rtrim('\0');
  }

  // It is not a long name so trim the blanks at the end of the name.
  if (Name[Name.size() - 1] != '/')
    return Name.rtrim(' ');

  // It's a simple name.
  return Name.drop_back(1);
}

Expected<uint32_t> ArchiveMemberHeader::getSize() const {
  uint32_t Ret;
  if (llvm::StringRef(ArMemHdr->Size,
        sizeof(ArMemHdr->Size)).rtrim(" ").getAsInteger(10, Ret)) {
    std::string Buf;
    raw_string_ostream OS(Buf);
    OS.write_escaped(llvm::StringRef(ArMemHdr->Size,
                                     sizeof(ArMemHdr->Size)).rtrim(" "));
    OS.flush();
    uint64_t Offset = reinterpret_cast<const char *>(ArMemHdr) -
                      Parent->getData().data();
    return malformedError("characters in size field in archive header are not "
                          "all decimal numbers: '" + Buf + "' for archive "
                          "member header at offset " + Twine(Offset));
  }
  return Ret;
}

sys::fs::perms ArchiveMemberHeader::getAccessMode() const {
  unsigned Ret;
  if (StringRef(ArMemHdr->AccessMode,
                sizeof(ArMemHdr->AccessMode)).rtrim(' ').getAsInteger(8, Ret))
    llvm_unreachable("Access mode is not an octal number.");
  return static_cast<sys::fs::perms>(Ret);
}

sys::TimeValue ArchiveMemberHeader::getLastModified() const {
  unsigned Seconds;
  if (StringRef(ArMemHdr->LastModified,
                sizeof(ArMemHdr->LastModified)).rtrim(' ')
          .getAsInteger(10, Seconds))
    llvm_unreachable("Last modified time not a decimal number.");

  sys::TimeValue Ret;
  Ret.fromEpochTime(Seconds);
  return Ret;
}

unsigned ArchiveMemberHeader::getUID() const {
  unsigned Ret;
  StringRef User = StringRef(ArMemHdr->UID, sizeof(ArMemHdr->UID)).rtrim(' ');
  if (User.empty())
    return 0;
  if (User.getAsInteger(10, Ret))
    llvm_unreachable("UID time not a decimal number.");
  return Ret;
}

unsigned ArchiveMemberHeader::getGID() const {
  unsigned Ret;
  StringRef Group = StringRef(ArMemHdr->GID, sizeof(ArMemHdr->GID)).rtrim(' ');
  if (Group.empty())
    return 0;
  if (Group.getAsInteger(10, Ret))
    llvm_unreachable("GID time not a decimal number.");
  return Ret;
}

Archive::Child::Child(const Archive *Parent, StringRef Data,
                      uint16_t StartOfFile)
    : Parent(Parent), Header(Parent, Data.data(), Data.size(), nullptr),
      Data(Data), StartOfFile(StartOfFile) {
}

Archive::Child::Child(const Archive *Parent, const char *Start, Error *Err)
    : Parent(Parent), Header(Parent, Start, Parent->getData().size() -
                             (Start - Parent->getData().data()), Err) {
  if (!Start)
    return;
  ErrorAsOutParameter ErrAsOutParam(Err);

  // If there was an error in the construction of the Header and we were passed
  // Err that is not nullptr then just return with the error now set.
  if (Err && *Err)
    return;

  uint64_t Size = Header.getSizeOf();
  Data = StringRef(Start, Size);
  Expected<bool> isThinOrErr = isThinMember();
  if (!isThinOrErr) {
    if (Err)
      *Err = isThinOrErr.takeError();
    return;
  }
  bool isThin = isThinOrErr.get();
  if (!isThin) {
    Expected<uint64_t> MemberSize = getRawSize();
    if (!MemberSize) {
      if (Err)
        *Err = MemberSize.takeError();
      return;
    }
    Size += MemberSize.get();
    Data = StringRef(Start, Size);
  }

  // Setup StartOfFile and PaddingBytes.
  StartOfFile = Header.getSizeOf();
  // Don't include attached name.
  Expected<StringRef> NameOrErr = getRawName();
  if (!NameOrErr){
    if (Err)
      *Err = NameOrErr.takeError();
    return;
  }
  StringRef Name = NameOrErr.get();
  if (Name.startswith("#1/")) {
    uint64_t NameSize;
    if (Name.substr(3).rtrim(' ').getAsInteger(10, NameSize)) {
      if (Err) {
        std::string Buf;
        raw_string_ostream OS(Buf);
        OS.write_escaped(Name.substr(3).rtrim(' '));
        OS.flush();
        uint64_t Offset = Start - Parent->getData().data();
        *Err = malformedError("long name length characters after the #1/ are "
                              "not all decimal numbers: '" + Buf + "' for "
                              "archive member header at offset " +
                              Twine(Offset));
        return;
      }
    }
    StartOfFile += NameSize;
  }
}

Expected<uint64_t> Archive::Child::getSize() const {
  if (Parent->IsThin) {
    Expected<uint32_t> Size = Header.getSize();
    if (!Size)
      return Size.takeError();
    return Size.get();
  }
  return Data.size() - StartOfFile;
}

Expected<uint64_t> Archive::Child::getRawSize() const {
  return Header.getSize();
}

Expected<bool> Archive::Child::isThinMember() const {
  Expected<StringRef> NameOrErr = Header.getRawName();
  if (!NameOrErr)
    return NameOrErr.takeError();
  StringRef Name = NameOrErr.get();
  return Parent->IsThin && Name != "/" && Name != "//";
}

ErrorOr<std::string> Archive::Child::getFullName() const {
  Expected<bool> isThin = isThinMember();
  if (!isThin)
    return errorToErrorCode(isThin.takeError());
  assert(isThin.get());
  Expected<StringRef> NameOrErr = getName();
  if (!NameOrErr)
    return errorToErrorCode(NameOrErr.takeError());
  StringRef Name = *NameOrErr;
  if (sys::path::is_absolute(Name))
    return Name;

  SmallString<128> FullName = sys::path::parent_path(
      Parent->getMemoryBufferRef().getBufferIdentifier());
  sys::path::append(FullName, Name);
  return StringRef(FullName);
}

ErrorOr<StringRef> Archive::Child::getBuffer() const {
  Expected<bool> isThinOrErr = isThinMember();
  if (!isThinOrErr)
    return errorToErrorCode(isThinOrErr.takeError());
  bool isThin = isThinOrErr.get();
  if (!isThin) {
    Expected<uint32_t> Size = getSize();
    if (!Size)
      return errorToErrorCode(Size.takeError());
    return StringRef(Data.data() + StartOfFile, Size.get());
  }
  ErrorOr<std::string> FullNameOrEr = getFullName();
  if (std::error_code EC = FullNameOrEr.getError())
    return EC;
  const std::string &FullName = *FullNameOrEr;
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(FullName);
  if (std::error_code EC = Buf.getError())
    return EC;
  Parent->ThinBuffers.push_back(std::move(*Buf));
  return Parent->ThinBuffers.back()->getBuffer();
}

Expected<Archive::Child> Archive::Child::getNext() const {
  size_t SpaceToSkip = Data.size();
  // If it's odd, add 1 to make it even.
  if (SpaceToSkip & 1)
    ++SpaceToSkip;

  const char *NextLoc = Data.data() + SpaceToSkip;

  // Check to see if this is at the end of the archive.
  if (NextLoc == Parent->Data.getBufferEnd())
    return Child(Parent, nullptr, nullptr);

  // Check to see if this is past the end of the archive.
  if (NextLoc > Parent->Data.getBufferEnd()) {
    std::string Msg("offset to next archive member past the end of the archive "
                    "after member ");
    Expected<StringRef> NameOrErr = getName();
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      uint64_t Offset = Data.data() - Parent->getData().data();
      return malformedError(Msg + "at offset " + Twine(Offset));
    } else
      return malformedError(Msg + NameOrErr.get());
  }

  Error Err;
  Child Ret(Parent, NextLoc, &Err);
  if (Err)
    return std::move(Err);
  return Ret;
}

uint64_t Archive::Child::getChildOffset() const {
  const char *a = Parent->Data.getBuffer().data();
  const char *c = Data.data();
  uint64_t offset = c - a;
  return offset;
}

Expected<StringRef> Archive::Child::getName() const {
  Expected<uint64_t> RawSizeOrErr = getRawSize();
  if (!RawSizeOrErr)
    return RawSizeOrErr.takeError();
  uint64_t RawSize = RawSizeOrErr.get();
  Expected<StringRef> NameOrErr = Header.getName(Header.getSizeOf() + RawSize);
  if (!NameOrErr)
    return NameOrErr.takeError();
  StringRef Name = NameOrErr.get();
  return Name;
}

Expected<MemoryBufferRef> Archive::Child::getMemoryBufferRef() const {
  Expected<StringRef> NameOrErr = getName();
  if (!NameOrErr)
    return NameOrErr.takeError();
  StringRef Name = NameOrErr.get();
  ErrorOr<StringRef> Buf = getBuffer();
  if (std::error_code EC = Buf.getError())
    return errorCodeToError(EC);
  return MemoryBufferRef(*Buf, Name);
}

Expected<std::unique_ptr<Binary>>
Archive::Child::getAsBinary(LLVMContext *Context) const {
  Expected<MemoryBufferRef> BuffOrErr = getMemoryBufferRef();
  if (!BuffOrErr)
    return BuffOrErr.takeError();

  auto BinaryOrErr = createBinary(BuffOrErr.get(), Context);
  if (BinaryOrErr)
    return std::move(*BinaryOrErr);
  return BinaryOrErr.takeError();
}

Expected<std::unique_ptr<Archive>> Archive::create(MemoryBufferRef Source) {
  Error Err;
  std::unique_ptr<Archive> Ret(new Archive(Source, Err));
  if (Err)
    return std::move(Err);
  return std::move(Ret);
}

void Archive::setFirstRegular(const Child &C) {
  FirstRegularData = C.Data;
  FirstRegularStartOfFile = C.StartOfFile;
}

Archive::Archive(MemoryBufferRef Source, Error &Err)
    : Binary(Binary::ID_Archive, Source) {
  ErrorAsOutParameter ErrAsOutParam(&Err);
  StringRef Buffer = Data.getBuffer();
  // Check for sufficient magic.
  if (Buffer.startswith(ThinMagic)) {
    IsThin = true;
  } else if (Buffer.startswith(Magic)) {
    IsThin = false;
  } else {
    Err = make_error<GenericBinaryError>("File too small to be an archive",
                                         object_error::invalid_file_type);
    return;
  }

  // Make sure Format is initialized before any call to
  // ArchiveMemberHeader::getName() is made.  This could be a valid empty
  // archive which is the same in all formats.  So claiming it to be gnu to is
  // fine if not totally correct before we look for a string table or table of
  // contents.
  Format = K_GNU;

  // Get the special members.
  child_iterator I = child_begin(Err, false);
  if (Err)
    return;
  child_iterator E = child_end();

  // See if this is a valid empty archive and if so return.
  if (I == E) {
    Err = Error::success();
    return;
  }
  const Child *C = &*I;

  auto Increment = [&]() {
    ++I;
    if (Err)
      return true;
    C = &*I;
    return false;
  };

  Expected<StringRef> NameOrErr = C->getRawName();
  if (!NameOrErr) {
    Err = NameOrErr.takeError();
    return;
  }
  StringRef Name = NameOrErr.get();

  // Below is the pattern that is used to figure out the archive format
  // GNU archive format
  //  First member : / (may exist, if it exists, points to the symbol table )
  //  Second member : // (may exist, if it exists, points to the string table)
  //  Note : The string table is used if the filename exceeds 15 characters
  // BSD archive format
  //  First member : __.SYMDEF or "__.SYMDEF SORTED" (the symbol table)
  //  There is no string table, if the filename exceeds 15 characters or has a
  //  embedded space, the filename has #1/<size>, The size represents the size
  //  of the filename that needs to be read after the archive header
  // COFF archive format
  //  First member : /
  //  Second member : / (provides a directory of symbols)
  //  Third member : // (may exist, if it exists, contains the string table)
  //  Note: Microsoft PE/COFF Spec 8.3 says that the third member is present
  //  even if the string table is empty. However, lib.exe does not in fact
  //  seem to create the third member if there's no member whose filename
  //  exceeds 15 characters. So the third member is optional.

  if (Name == "__.SYMDEF" || Name == "__.SYMDEF_64") {
    if (Name == "__.SYMDEF")
      Format = K_BSD;
    else // Name == "__.SYMDEF_64"
      Format = K_DARWIN64;
    // We know that the symbol table is not an external file, so we just assert
    // there is no error.
    SymbolTable = *C->getBuffer();
    if (Increment())
      return;
    setFirstRegular(*C);

    Err = Error::success();
    return;
  }

  if (Name.startswith("#1/")) {
    Format = K_BSD;
    // We know this is BSD, so getName will work since there is no string table.
    Expected<StringRef> NameOrErr = C->getName();
    if (!NameOrErr) {
      Err = NameOrErr.takeError();
      return;
    }
    Name = NameOrErr.get();
    if (Name == "__.SYMDEF SORTED" || Name == "__.SYMDEF") {
      // We know that the symbol table is not an external file, so we just
      // assert there is no error.
      SymbolTable = *C->getBuffer();
      if (Increment())
        return;
    }
    else if (Name == "__.SYMDEF_64 SORTED" || Name == "__.SYMDEF_64") {
      Format = K_DARWIN64;
      // We know that the symbol table is not an external file, so we just
      // assert there is no error.
      SymbolTable = *C->getBuffer();
      if (Increment())
        return;
    }
    setFirstRegular(*C);
    return;
  }

  // MIPS 64-bit ELF archives use a special format of a symbol table.
  // This format is marked by `ar_name` field equals to "/SYM64/".
  // For detailed description see page 96 in the following document:
  // http://techpubs.sgi.com/library/manuals/4000/007-4658-001/pdf/007-4658-001.pdf

  bool has64SymTable = false;
  if (Name == "/" || Name == "/SYM64/") {
    // We know that the symbol table is not an external file, so we just assert
    // there is no error.
    SymbolTable = *C->getBuffer();
    if (Name == "/SYM64/")
      has64SymTable = true;

    if (Increment())
      return;
    if (I == E) {
      Err = Error::success();
      return;
    }
    Expected<StringRef> NameOrErr = C->getRawName();
    if (!NameOrErr) {
      Err = NameOrErr.takeError();
      return;
    }
    Name = NameOrErr.get();
  }

  if (Name == "//") {
    Format = has64SymTable ? K_MIPS64 : K_GNU;
    // The string table is never an external member, so we just assert on the
    // ErrorOr.
    StringTable = *C->getBuffer();
    if (Increment())
      return;
    setFirstRegular(*C);
    Err = Error::success();
    return;
  }

  if (Name[0] != '/') {
    Format = has64SymTable ? K_MIPS64 : K_GNU;
    setFirstRegular(*C);
    Err = Error::success();
    return;
  }

  if (Name != "/") {
    Err = errorCodeToError(object_error::parse_failed);
    return;
  }

  Format = K_COFF;
  // We know that the symbol table is not an external file, so we just assert
  // there is no error.
  SymbolTable = *C->getBuffer();

  if (Increment())
    return;

  if (I == E) {
    setFirstRegular(*C);
    Err = Error::success();
    return;
  }

  NameOrErr = C->getRawName();
  if (!NameOrErr) {
    Err = NameOrErr.takeError();
    return;
  }
  Name = NameOrErr.get();

  if (Name == "//") {
    // The string table is never an external member, so we just assert on the
    // ErrorOr.
    StringTable = *C->getBuffer();
    if (Increment())
      return;
  }

  setFirstRegular(*C);
  Err = Error::success();
}

Archive::child_iterator Archive::child_begin(Error &Err,
                                             bool SkipInternal) const {
  if (Data.getBufferSize() == 8) // empty archive.
    return child_end();

  if (SkipInternal)
    return child_iterator(Child(this, FirstRegularData,
                                FirstRegularStartOfFile),
                          &Err);

  const char *Loc = Data.getBufferStart() + strlen(Magic);
  Child C(this, Loc, &Err);
  if (Err)
    return child_end();
  return child_iterator(C, &Err);
}

Archive::child_iterator Archive::child_end() const {
  return child_iterator(Child(this, nullptr, nullptr), nullptr);
}

StringRef Archive::Symbol::getName() const {
  return Parent->getSymbolTable().begin() + StringIndex;
}

ErrorOr<Archive::Child> Archive::Symbol::getMember() const {
  const char *Buf = Parent->getSymbolTable().begin();
  const char *Offsets = Buf;
  if (Parent->kind() == K_MIPS64 || Parent->kind() == K_DARWIN64)
    Offsets += sizeof(uint64_t);
  else
    Offsets += sizeof(uint32_t);
  uint32_t Offset = 0;
  if (Parent->kind() == K_GNU) {
    Offset = read32be(Offsets + SymbolIndex * 4);
  } else if (Parent->kind() == K_MIPS64) {
    Offset = read64be(Offsets + SymbolIndex * 8);
  } else if (Parent->kind() == K_BSD) {
    // The SymbolIndex is an index into the ranlib structs that start at
    // Offsets (the first uint32_t is the number of bytes of the ranlib
    // structs).  The ranlib structs are a pair of uint32_t's the first
    // being a string table offset and the second being the offset into
    // the archive of the member that defines the symbol.  Which is what
    // is needed here.
    Offset = read32le(Offsets + SymbolIndex * 8 + 4);
  } else if (Parent->kind() == K_DARWIN64) {
    // The SymbolIndex is an index into the ranlib_64 structs that start at
    // Offsets (the first uint64_t is the number of bytes of the ranlib_64
    // structs).  The ranlib_64 structs are a pair of uint64_t's the first
    // being a string table offset and the second being the offset into
    // the archive of the member that defines the symbol.  Which is what
    // is needed here.
    Offset = read64le(Offsets + SymbolIndex * 16 + 8);
  } else {
    // Skip offsets.
    uint32_t MemberCount = read32le(Buf);
    Buf += MemberCount * 4 + 4;

    uint32_t SymbolCount = read32le(Buf);
    if (SymbolIndex >= SymbolCount)
      return object_error::parse_failed;

    // Skip SymbolCount to get to the indices table.
    const char *Indices = Buf + 4;

    // Get the index of the offset in the file member offset table for this
    // symbol.
    uint16_t OffsetIndex = read16le(Indices + SymbolIndex * 2);
    // Subtract 1 since OffsetIndex is 1 based.
    --OffsetIndex;

    if (OffsetIndex >= MemberCount)
      return object_error::parse_failed;

    Offset = read32le(Offsets + OffsetIndex * 4);
  }

  const char *Loc = Parent->getData().begin() + Offset;
  Error Err;
  Child C(Parent, Loc, &Err);
  if (Err)
    return errorToErrorCode(std::move(Err));
  return C;
}

Archive::Symbol Archive::Symbol::getNext() const {
  Symbol t(*this);
  if (Parent->kind() == K_BSD) {
    // t.StringIndex is an offset from the start of the __.SYMDEF or
    // "__.SYMDEF SORTED" member into the string table for the ranlib
    // struct indexed by t.SymbolIndex .  To change t.StringIndex to the
    // offset in the string table for t.SymbolIndex+1 we subtract the
    // its offset from the start of the string table for t.SymbolIndex
    // and add the offset of the string table for t.SymbolIndex+1.

    // The __.SYMDEF or "__.SYMDEF SORTED" member starts with a uint32_t
    // which is the number of bytes of ranlib structs that follow.  The ranlib
    // structs are a pair of uint32_t's the first being a string table offset
    // and the second being the offset into the archive of the member that
    // define the symbol. After that the next uint32_t is the byte count of
    // the string table followed by the string table.
    const char *Buf = Parent->getSymbolTable().begin();
    uint32_t RanlibCount = 0;
    RanlibCount = read32le(Buf) / 8;
    // If t.SymbolIndex + 1 will be past the count of symbols (the RanlibCount)
    // don't change the t.StringIndex as we don't want to reference a ranlib
    // past RanlibCount.
    if (t.SymbolIndex + 1 < RanlibCount) {
      const char *Ranlibs = Buf + 4;
      uint32_t CurRanStrx = 0;
      uint32_t NextRanStrx = 0;
      CurRanStrx = read32le(Ranlibs + t.SymbolIndex * 8);
      NextRanStrx = read32le(Ranlibs + (t.SymbolIndex + 1) * 8);
      t.StringIndex -= CurRanStrx;
      t.StringIndex += NextRanStrx;
    }
  } else {
    // Go to one past next null.
    t.StringIndex = Parent->getSymbolTable().find('\0', t.StringIndex) + 1;
  }
  ++t.SymbolIndex;
  return t;
}

Archive::symbol_iterator Archive::symbol_begin() const {
  if (!hasSymbolTable())
    return symbol_iterator(Symbol(this, 0, 0));

  const char *buf = getSymbolTable().begin();
  if (kind() == K_GNU) {
    uint32_t symbol_count = 0;
    symbol_count = read32be(buf);
    buf += sizeof(uint32_t) + (symbol_count * (sizeof(uint32_t)));
  } else if (kind() == K_MIPS64) {
    uint64_t symbol_count = read64be(buf);
    buf += sizeof(uint64_t) + (symbol_count * (sizeof(uint64_t)));
  } else if (kind() == K_BSD) {
    // The __.SYMDEF or "__.SYMDEF SORTED" member starts with a uint32_t
    // which is the number of bytes of ranlib structs that follow.  The ranlib
    // structs are a pair of uint32_t's the first being a string table offset
    // and the second being the offset into the archive of the member that
    // define the symbol. After that the next uint32_t is the byte count of
    // the string table followed by the string table.
    uint32_t ranlib_count = 0;
    ranlib_count = read32le(buf) / 8;
    const char *ranlibs = buf + 4;
    uint32_t ran_strx = 0;
    ran_strx = read32le(ranlibs);
    buf += sizeof(uint32_t) + (ranlib_count * (2 * (sizeof(uint32_t))));
    // Skip the byte count of the string table.
    buf += sizeof(uint32_t);
    buf += ran_strx;
  } else if (kind() == K_DARWIN64) {
    // The __.SYMDEF_64 or "__.SYMDEF_64 SORTED" member starts with a uint64_t
    // which is the number of bytes of ranlib_64 structs that follow.  The
    // ranlib_64 structs are a pair of uint64_t's the first being a string
    // table offset and the second being the offset into the archive of the
    // member that define the symbol. After that the next uint64_t is the byte
    // count of the string table followed by the string table.
    uint64_t ranlib_count = 0;
    ranlib_count = read64le(buf) / 16;
    const char *ranlibs = buf + 8;
    uint64_t ran_strx = 0;
    ran_strx = read64le(ranlibs);
    buf += sizeof(uint64_t) + (ranlib_count * (2 * (sizeof(uint64_t))));
    // Skip the byte count of the string table.
    buf += sizeof(uint64_t);
    buf += ran_strx;
  } else {
    uint32_t member_count = 0;
    uint32_t symbol_count = 0;
    member_count = read32le(buf);
    buf += 4 + (member_count * 4); // Skip offsets.
    symbol_count = read32le(buf);
    buf += 4 + (symbol_count * 2); // Skip indices.
  }
  uint32_t string_start_offset = buf - getSymbolTable().begin();
  return symbol_iterator(Symbol(this, 0, string_start_offset));
}

Archive::symbol_iterator Archive::symbol_end() const {
  return symbol_iterator(Symbol(this, getNumberOfSymbols(), 0));
}

uint32_t Archive::getNumberOfSymbols() const {
  if (!hasSymbolTable())
    return 0;
  const char *buf = getSymbolTable().begin();
  if (kind() == K_GNU)
    return read32be(buf);
  if (kind() == K_MIPS64)
    return read64be(buf);
  if (kind() == K_BSD)
    return read32le(buf) / 8;
  if (kind() == K_DARWIN64)
    return read64le(buf) / 16;
  uint32_t member_count = 0;
  member_count = read32le(buf);
  buf += 4 + (member_count * 4); // Skip offsets.
  return read32le(buf);
}

Expected<Optional<Archive::Child>> Archive::findSym(StringRef name) const {
  Archive::symbol_iterator bs = symbol_begin();
  Archive::symbol_iterator es = symbol_end();

  for (; bs != es; ++bs) {
    StringRef SymName = bs->getName();
    if (SymName == name) {
      if (auto MemberOrErr = bs->getMember())
        return Child(*MemberOrErr);
      else
        return errorCodeToError(MemberOrErr.getError());
    }
  }
  return Optional<Child>();
}

bool Archive::hasSymbolTable() const { return !SymbolTable.empty(); }
