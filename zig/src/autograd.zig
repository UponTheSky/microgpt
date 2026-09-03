const std = @import("std");
const ArrayList = std.ArrayList;

pub const Value = struct {
    data: f32,
    grad: f32,
    children: ArrayList(*Value),
    localGrads: ArrayList(f32),
};

pub fn new(data: f32, children: ArrayList(*Value), localGrads: [*]f32) Value {
    return .{
        .data = data,
        .grad = 0,
        .children = children,
        .localGrads = localGrads,
    };
}

pub fn add(lhs: Value, rhs: Value) Value {
    return new(lhs.data + rhs.data, .{ &rhs, &lhs }, .{ 1, 1 });
}

pub fn mul(lhs: Value, rhs: Value) Value {
    return new(lhs.data * rhs.data, .{ &lhs, &rhs }, .{ rhs.data, lhs.data });
}

pub fn pow(lhs: Value, rhs: f32) Value {
    return new(std.math.pow(lhs.data, rhs), .{&lhs}, .{rhs * std.math.pow(lhs.data, rhs - 1)});
}

pub fn log(val: Value) Value {
    return new(std.math.log(val.data), .{&val}, .{1 / val.data});
}

pub fn exp(val: Value) Value {
    return new(std.math.exp(val.data), .{&val}, .{std.math.exp(val.data)});
}

pub fn relu(val: Value) Value {
    return new(@max(0, val.data), .{&val}, .{if (val.data > 0) 1 else 0});
}

pub fn neg(val: Value) Value {
    return mul(val, -1);
}

pub fn sub(lhs: Value, rhs: Value) Value {
    return add(lhs, neg(rhs));
}

pub fn div(lhs: Value, rhs: Value) Value {
    return mul(lhs, pow(rhs, -1));
}

fn buildTopo(v: *Value, topo: *ArrayList(*Value), visited: *std.AutoHashMap(*Value, void)) void {
    if (!visited.contains(v)) {
        visited.put(v, void);

        for (v.children) |child| {
            buildTopo(child);
        }

        topo.append(v);
    }
}

pub fn backward(val: *Value) void {
    var topo: ArrayList(*Value) = .empty;

    var visited: std.AutoHashMap(*Value, void) = .empty;

    buildTopo(val, &topo, &visited);

    val.grad = 1;

    std.mem.reverse(*Value, topo.items);

    for (topo) |v| {
        for (v.children, v.localGrads) |child, localGrad| {
            child.grad += localGrad * v.grad;
        }
    }
}
