"""Ordered backtracking bytecode interpreter used by the Python bindings."""

from std.algorithm import parallelize
from std.sys import simd_width_of

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt32, AnyOrigin[mut=True]]

comptime OP_MATCH = 0
comptime OP_CHAR = 1
comptime OP_ANY = 2
comptime OP_CLASS = 3
comptime OP_JMP = 4
comptime OP_SPLIT = 5
comptime OP_SAVE = 6
comptime OP_BOL = 7
comptime OP_EOL = 8
comptime OP_BACKREF = 9
comptime OP_BOUNDARY = 10
comptime OP_NOT_BOUNDARY = 11
comptime OP_BEGIN = 12
comptime OP_END = 13
comptime OP_MARK = 14
comptime OP_REPEAT_CLASS = 15
comptime OP_REPEAT_BOUNDS = 16
comptime OP_REPEAT_CHAR = 17

comptime FLAG_IGNORECASE = 2
comptime FLAG_MULTILINE = 8
comptime FLAG_DOTALL = 16


def ascii_lower(c: Int) -> Int:
    if c >= 65 and c <= 90:
        return c + 32
    return c


def is_word(c: Int) -> Bool:
    return (c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (
        c >= 97 and c <= 122
    ) or c == 95


def chars_equal(a: Int, b: Int, flags: Int) -> Bool:
    if (flags & FLAG_IGNORECASE) != 0:
        return ascii_lower(a) == ascii_lower(b)
    return a == b


def find_prefix_serial(
    text: UPtr, start: Int, stop: Int, prefix: Int, flags: Int
) -> Int:
    var i = start
    if (flags & FLAG_IGNORECASE) != 0:
        while i < stop:
            if chars_equal(Int(text[i]), prefix, flags):
                return i
            i += 1
        return stop

    comptime W = simd_width_of[DType.float64]()
    var target = SIMD[DType.uint32, W](UInt32(prefix))
    var ones = SIMD[DType.uint32, W](1)
    var zeros = SIMD[DType.uint32, W](0)
    while i + W <= stop:
        var values = text.load[width=W](i)
        var matches = values.eq(target).select(ones, zeros)
        if Int(matches.reduce_add()[0]) != 0:
            for lane in range(W):
                if Int(values[lane]) == prefix:
                    return i + lane
        i += W
    while i < stop:
        if Int(text[i]) == prefix:
            return i
        i += 1
    return stop


def find_prefix(
    text: UPtr,
    start: Int,
    stop: Int,
    prefix: Int,
    flags: Int,
    results: IPtr,
) -> Int:
    comptime PARALLEL_THRESHOLD = 4_000_000
    comptime TASKS = 8
    if stop - start < PARALLEL_THRESHOLD or (
        flags & FLAG_IGNORECASE
    ) != 0:
        return find_prefix_serial(text, start, stop, prefix, flags)

    var chunk = (stop - start + TASKS - 1) // TASKS

    @parameter
    def scan_chunk(index: Int):
        var first = start + index * chunk
        var last = min(stop, first + chunk)
        if first < stop:
            results[index] = Int64(
                find_prefix_serial(text, first, last, prefix, flags)
            )
        else:
            results[index] = Int64(stop)

    parallelize[scan_chunk](TASKS, TASKS)
    var found = stop
    for index in range(TASKS):
        found = min(found, Int(results[index]))
    return found


def class_matches(
    value: Int, ranges: IPtr, offset: Int, encoded_count: Int
) -> Bool:
    var count = encoded_count
    var negated = False
    if count < 0:
        count = -count - 1
        negated = True
    var inside = False
    for ri in range(count):
        var lo = Int(ranges[(offset + ri) * 2])
        var hi = Int(ranges[(offset + ri) * 2 + 1])
        if value >= lo and value <= hi:
            inside = True
            break
    return not inside if negated else inside


def class_vector_matches(
    text: UPtr,
    position: Int,
    ranges: IPtr,
    offset: Int,
    encoded_count: Int,
) -> Bool:
    comptime W = simd_width_of[DType.float64]()
    var values = text.load[width=W](position)
    var inside = SIMD[DType.bool, W](fill=False)
    var count = encoded_count
    var negated = False
    if count < 0:
        count = -count - 1
        negated = True
    for ri in range(count):
        var lo = SIMD[DType.uint32, W](
            UInt32(ranges[(offset + ri) * 2])
        )
        var hi = SIMD[DType.uint32, W](
            UInt32(ranges[(offset + ri) * 2 + 1])
        )
        inside = inside | (values.ge(lo) & values.le(hi))
    if negated:
        inside = ~inside
    var matches = inside.select(
        SIMD[DType.uint32, W](1), SIMD[DType.uint32, W](0)
    )
    return Int(matches.reduce_add()[0]) == W


def char_vector_matches(
    text: UPtr, position: Int, value: Int, flags: Int
) -> Bool:
    comptime W = simd_width_of[DType.float64]()
    if (flags & FLAG_IGNORECASE) != 0:
        return False
    var values = text.load[width=W](position)
    var expected = SIMD[DType.uint32, W](UInt32(value))
    var matches = values.eq(expected).select(
        SIMD[DType.uint32, W](1), SIMD[DType.uint32, W](0)
    )
    return Int(matches.reduce_add()[0]) == W


@export("mrx_exec")
def mrx_exec(
    code_addr: Int,
    code_len: Int,
    ranges_addr: Int,
    text_addr: Int,
    text_len: Int,
    start_pos: Int,
    end_pos: Int,
    mode: Int,
    flags: Int,
    prefix: Int,
    captures_addr: Int,
    capture_count: Int,
    stack_pc_addr: Int,
    stack_pos_addr: Int,
    stack_log_addr: Int,
    stack_min_addr: Int,
    stack_capacity: Int,
    log_slot_addr: Int,
    log_old_addr: Int,
    log_capacity: Int,
) abi("C") -> Int:
    if code_addr == 0 or ranges_addr == 0 or text_addr == 0 or (
        captures_addr == 0 or stack_pc_addr == 0 or stack_pos_addr == 0
    ) or (
        stack_log_addr == 0 or stack_min_addr == 0 or log_slot_addr == 0
    ) or log_old_addr == 0:
        return -4
    if code_len <= 0 or text_len < 0 or start_pos < 0 or (
        start_pos > text_len or end_pos < 0 or end_pos > text_len
    ) or mode < 0 or mode > 3 or prefix < -1 or prefix > 0x10FFFF:
        return -4
    if capture_count < 4 or capture_count % 2 != 0 or (
        stack_capacity < 8 or log_capacity <= 0
    ):
        return -4

    var code = IPtr(unsafe_from_address=code_addr)
    var ranges = IPtr(unsafe_from_address=ranges_addr)
    var text = UPtr(unsafe_from_address=text_addr)
    var captures = IPtr(unsafe_from_address=captures_addr)
    var stack_pc = IPtr(unsafe_from_address=stack_pc_addr)
    var stack_pos = IPtr(unsafe_from_address=stack_pos_addr)
    var stack_log = IPtr(unsafe_from_address=stack_log_addr)
    var stack_min = IPtr(unsafe_from_address=stack_min_addr)
    var log_slot = IPtr(unsafe_from_address=log_slot_addr)
    var log_old = IPtr(unsafe_from_address=log_old_addr)

    var candidate = start_pos
    while candidate <= end_pos:
        if (mode == 1 or mode == 2) and candidate != start_pos:
            return 0
        if prefix >= 0:
            if mode == 1 or mode == 2:
                if candidate >= end_pos or not chars_equal(
                    Int(text[candidate]), prefix, flags
                ):
                    return 0
            else:
                candidate = find_prefix(
                    text, candidate, end_pos, prefix, flags, stack_min
                )
                if candidate >= end_pos:
                    return 0

        for i in range(capture_count):
            captures[i] = -1

        var pc = 0
        var pos = candidate
        var stack_size = 0
        var log_size = 0
        var running = True

        while running:
            if pc < 0 or pc >= code_len:
                running = False
            else:
                var base = pc * 3
                var op = Int(code[base])
                var a = Int(code[base + 1])
                var b = Int(code[base + 2])
                var failed = False

                if op == OP_MATCH:
                    if (mode != 2 or pos == end_pos) and (
                        mode != 3 or pos > candidate
                    ):
                        return 1
                    failed = True
                elif op == OP_CHAR:
                    if pos < end_pos and chars_equal(Int(text[pos]), a, flags):
                        pos += 1
                        pc += 1
                    else:
                        failed = True
                elif op == OP_ANY:
                    if pos < end_pos and (
                        (flags & FLAG_DOTALL) != 0 or Int(text[pos]) != 10
                    ):
                        pos += 1
                        pc += 1
                    else:
                        failed = True
                elif op == OP_CLASS:
                    if pos >= end_pos:
                        failed = True
                    else:
                        if class_matches(Int(text[pos]), ranges, a, b):
                            pos += 1
                            pc += 1
                        else:
                            failed = True
                elif op == OP_JMP:
                    pc = a
                elif op == OP_SPLIT:
                    if stack_size >= stack_capacity:
                        return -2
                    stack_pc[stack_size] = Int64(b)
                    stack_pos[stack_size] = Int64(pos)
                    stack_log[stack_size] = Int64(log_size)
                    stack_min[stack_size] = -1
                    stack_size += 1
                    pc = a
                elif op == OP_SAVE:
                    if log_size >= log_capacity:
                        return -2
                    log_slot[log_size] = Int64(a)
                    log_old[log_size] = captures[a]
                    log_size += 1
                    captures[a] = Int64(pos)
                    pc += 1
                elif op == OP_BOL:
                    if pos == 0 or (
                        (flags & FLAG_MULTILINE) != 0
                        and pos > 0
                        and Int(text[pos - 1]) == 10
                    ):
                        pc += 1
                    else:
                        failed = True
                elif op == OP_EOL:
                    if pos == end_pos or (
                        pos + 1 == end_pos and Int(text[pos]) == 10
                    ) or (
                        (flags & FLAG_MULTILINE) != 0
                        and pos < end_pos
                        and Int(text[pos]) == 10
                    ):
                        pc += 1
                    else:
                        failed = True
                elif op == OP_BACKREF:
                    var s = Int(captures[a * 2])
                    var e = Int(captures[a * 2 + 1])
                    if s < 0 or e < s or pos + (e - s) > end_pos:
                        failed = True
                    else:
                        var length = e - s
                        var same = True
                        comptime W = simd_width_of[DType.float64]()
                        var j = 0
                        if (flags & FLAG_IGNORECASE) == 0:
                            while j + W <= length:
                                var left = text.load[width=W](s + j)
                                var right = text.load[width=W](pos + j)
                                var equal = left.eq(right).select(
                                    SIMD[DType.uint32, W](1),
                                    SIMD[DType.uint32, W](0),
                                )
                                if Int(equal.reduce_add()[0]) != W:
                                    same = False
                                    break
                                j += W
                        while same and j < length:
                            if not chars_equal(
                                Int(text[s + j]), Int(text[pos + j]), flags
                            ):
                                same = False
                                break
                            j += 1
                        if same:
                            pos += length
                            pc += 1
                        else:
                            failed = True
                elif op == OP_BOUNDARY or op == OP_NOT_BOUNDARY:
                    var left = pos > 0 and is_word(Int(text[pos - 1]))
                    var right = pos < end_pos and is_word(Int(text[pos]))
                    var boundary = left != right
                    if (op == OP_BOUNDARY and boundary) or (
                        op == OP_NOT_BOUNDARY and not boundary
                    ):
                        pc += 1
                    else:
                        failed = True
                elif op == OP_BEGIN:
                    if pos == 0:
                        pc += 1
                    else:
                        failed = True
                elif op == OP_END:
                    if pos == end_pos:
                        pc += 1
                    else:
                        failed = True
                elif op == OP_MARK:
                    if log_size + 1 >= log_capacity:
                        return -2
                    var index_slot = capture_count - 2
                    log_slot[log_size] = Int64(index_slot)
                    log_old[log_size] = captures[index_slot]
                    log_size += 1
                    captures[index_slot] = Int64(a)
                    if b != 0:
                        var name_slot = capture_count - 1
                        log_slot[log_size] = Int64(name_slot)
                        log_old[log_size] = captures[name_slot]
                        log_size += 1
                        captures[name_slot] = Int64(a)
                    pc += 1
                elif op == OP_REPEAT_CLASS:
                    var minimum = Int(code[(pc + 1) * 3 + 1])
                    var repeat_start = pos
                    var consumed = 0
                    comptime W = simd_width_of[DType.float64]()
                    while pos + W <= end_pos and class_vector_matches(
                        text, pos, ranges, a, b
                    ):
                        pos += W
                        consumed += W
                    while pos < end_pos and class_matches(
                        Int(text[pos]), ranges, a, b
                    ):
                        pos += 1
                        consumed += 1
                    if consumed < minimum:
                        failed = True
                    else:
                        pc += 2
                        var minimum_pos = repeat_start + minimum
                        if pos > minimum_pos:
                            if stack_size >= stack_capacity:
                                return -2
                            stack_pc[stack_size] = Int64(pc)
                            stack_pos[stack_size] = Int64(pos - 1)
                            stack_log[stack_size] = Int64(log_size)
                            stack_min[stack_size] = Int64(minimum_pos)
                            stack_size += 1
                elif op == OP_REPEAT_CHAR:
                    var minimum = Int(code[(pc + 1) * 3 + 1])
                    var repeat_start = pos
                    var consumed = 0
                    comptime W = simd_width_of[DType.float64]()
                    while pos + W <= end_pos and char_vector_matches(
                        text, pos, a, flags
                    ):
                        pos += W
                        consumed += W
                    while pos < end_pos and chars_equal(
                        Int(text[pos]), a, flags
                    ):
                        pos += 1
                        consumed += 1
                    if consumed < minimum:
                        failed = True
                    else:
                        pc += 2
                        var minimum_pos = repeat_start + minimum
                        if pos > minimum_pos:
                            if stack_size >= stack_capacity:
                                return -2
                            stack_pc[stack_size] = Int64(pc)
                            stack_pos[stack_size] = Int64(pos - 1)
                            stack_log[stack_size] = Int64(log_size)
                            stack_min[stack_size] = Int64(minimum_pos)
                            stack_size += 1
                elif op == OP_REPEAT_BOUNDS:
                    return -3
                else:
                    return -3

                if failed:
                    if stack_size == 0:
                        running = False
                    else:
                        stack_size -= 1
                        pc = Int(stack_pc[stack_size])
                        pos = Int(stack_pos[stack_size])
                        var target_log = Int(stack_log[stack_size])
                        while log_size > target_log:
                            log_size -= 1
                            var slot = Int(log_slot[log_size])
                            captures[slot] = log_old[log_size]
                        var minimum_pos = Int(stack_min[stack_size])
                        if minimum_pos >= 0 and pos > minimum_pos:
                            if stack_size >= stack_capacity:
                                return -2
                            stack_pc[stack_size] = Int64(pc)
                            stack_pos[stack_size] = Int64(pos - 1)
                            stack_log[stack_size] = Int64(log_size)
                            stack_min[stack_size] = Int64(minimum_pos)
                            stack_size += 1

        candidate += 1
    return 0


@export("mrx_find_all")
def mrx_find_all(
    code_addr: Int,
    code_len: Int,
    ranges_addr: Int,
    text_addr: Int,
    text_len: Int,
    start_pos: Int,
    end_pos: Int,
    mode: Int,
    flags: Int,
    prefix: Int,
    captures_addr: Int,
    capture_count: Int,
    stack_pc_addr: Int,
    stack_pos_addr: Int,
    stack_log_addr: Int,
    stack_min_addr: Int,
    stack_capacity: Int,
    log_slot_addr: Int,
    log_old_addr: Int,
    log_capacity: Int,
    results_addr: Int,
    result_capacity: Int,
    cursor_addr: Int,
    overlapped: Int,
) abi("C") -> Int:
    if captures_addr == 0 or results_addr == 0 or cursor_addr == 0 or (
        result_capacity <= 0
    ):
        return -4
    var captures = IPtr(unsafe_from_address=captures_addr)
    var results = IPtr(unsafe_from_address=results_addr)
    var cursor_ptr = IPtr(unsafe_from_address=cursor_addr)
    var cursor = Int(cursor_ptr[0])
    var span_slots = capture_count - 2
    var count = 0

    while cursor <= end_pos and count + 2 <= result_capacity:
        var status = mrx_exec(
            code_addr,
            code_len,
            ranges_addr,
            text_addr,
            text_len,
            cursor,
            end_pos,
            0,
            flags,
            prefix,
            captures_addr,
            capture_count,
            stack_pc_addr,
            stack_pos_addr,
            stack_log_addr,
            stack_min_addr,
            stack_capacity,
            log_slot_addr,
            log_old_addr,
            log_capacity,
        )
        if status <= 0:
            cursor = end_pos + 1
            if status < 0:
                return status
            break

        for slot in range(span_slots):
            results[count * span_slots + slot] = captures[slot]
        count += 1
        var match_start = Int(captures[0])
        var match_end = Int(captures[1])

        if overlapped != 0:
            cursor = match_start + 1
        elif match_end > match_start:
            cursor = match_end
        else:
            status = mrx_exec(
                code_addr,
                code_len,
                ranges_addr,
                text_addr,
                text_len,
                match_start,
                end_pos,
                3,
                flags,
                prefix,
                captures_addr,
                capture_count,
                stack_pc_addr,
                stack_pos_addr,
                stack_log_addr,
                stack_min_addr,
                stack_capacity,
                log_slot_addr,
                log_old_addr,
                log_capacity,
            )
            if status < 0:
                return status
            if status == 1 and Int(captures[0]) == match_start:
                for slot in range(span_slots):
                    results[count * span_slots + slot] = captures[slot]
                count += 1
                cursor = Int(captures[1])
            else:
                cursor = match_start + 1

    cursor_ptr[0] = Int64(cursor)
    return count
