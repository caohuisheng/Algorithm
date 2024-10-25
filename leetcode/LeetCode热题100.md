## 哈希

[1. 两数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> val2Index = new HashMap<>();
        // 遍历每个元素
        for(int i=0;i<nums.length;i++){
            int x = nums[i];
            // 如果之前遍历的元素存在值为target-x的，直接返回结果
            if(val2Index.containsKey(target - x)){
                return new int[]{i,val2Index.get(target - x)};
            }
            // 记录当前元素的值和下标
            val2Index.put(x,i);
        }
        return null;
    }
}
```

[49. 字母异位词分组 - 力扣（LeetCode）](https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-100-liked)

思路：题目要将字符串数组中的字母异位词（由重新排列源单词所有字母得到的一个新单词）组合在一起，并返回结果。如何判断两个单词是异位词呢，可以对单词进行编码，一共有26个字母，因此可以使用code[]数组统计每个字母的出现次数，将code数组转为字符串即为编码值。最后使用HashMap存储结果，键为对应异位词的编码值，值为编码值为对应编码值的所有字符串列表。

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        // 编码-分组映射
        Map<String,List<String>> code2Group=new HashMap<>();
        // 遍历每一种编码，将相同编码的元素添加到同一组中
        for(String str:strs){
            String code=encode(str);
            code2Group.putIfAbsent(code,new ArrayList<>());
            code2Group.get(code).add(str);
        }
        List<List<String>> res=new ArrayList<>();
        for(List<String> group:code2Group.values()){
            res.add(group);
        }
        return res;
    }
    //利用每个字符的出现次数进行编码
    String encode(String str){
        char[] code=new char[26];
        for(char c:str.toCharArray()){
            code[c-'a']++;
        }
        return new String(code);
    }
}
```

[128. 最长连续序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

- 将所有元素添加到集合set中，便于判断该元素是否存在。
- 遍历所有的元素，对于每个元素，循环查找以当前元素开始的连续元素，并记录连续序列的长度
- 设当前元素值为x，如果存在值为x-1的元素，直接跳过（不是连续序列的开始位置）

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        //将所有元素添加到集合，便于查看是否存在
        Set<Integer> set=new HashSet<>();
        for(int x:nums){
            set.add(x);
        }
        int res=0;
        //寻找每一个序列
        for(int x:set){
            // 存在比当前值x更小的，不是序列开始位置
            if(set.contains(x-1)) continue;
            int len=1;
            // 循环查找比当前元素大1的元素
            while(set.contains(x+1)){
                len++;
                x++;
            }
            res=Math.max(res,len);
        }
        return res;
    }
    
}
```

思路二：

将数组去重并排序，遍历数组，每次以当前位置i作为连续序列的起始位置，并记录连续序列的长度，再将指针移到该连续序列的末尾，开始查找下一个连续序列。

```java
class Solution {
	public int longestConsecutive(int[] nums) {
        //将数组去重
        Set<Integer> s = new HashSet<>();
        for(int x:nums){
            s.add(x);
        }
        Integer[] nums2 = s.toArray(new Integer[0]);
        //将数组排序
        Arrays.sort(nums2);
        int n = nums2.length;
        int i=0;
        int len = 0;
		//遍历每个元素
        while(i<n){
            int j=i;
            //查找以nums[i]为起点的连续序列
            while(j+1<n && nums2[j+1]-nums2[j]==1) j++;
            len = Math.max(len,j-i+1);
            //移到下一个连续序列的起始位置
            i = j+1;
        }
        return len;
    }
}
```

## 双指针

[283. 移动零 - 力扣（LeetCode）](https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int i = 0;
        // 将非0数字移到前面
        for(int j=0;j<nums.length;j++){
            if(nums[j] != 0){
                nums[i++] = nums[j];
            }
        }
        // 将后面的元素全部置0
        while(i<nums.length) nums[i++] = 0;
    }
}
```

[11. 盛最多水的容器 - 力扣（LeetCode）](https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int maxArea(int[] height) {
        int left=0,right=height.length-1;
        int res=0;
        while(left<right){
            int t=Math.min(height[left],height[right])*(right-left);
            res=Math.max(res,t);
            if(height[left]<height[right]) left++;
            else right--;
        }
        return res;
    }
}
```

[15. 三数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：首先将数组排序，枚举三元组的前2个元素的位置i, j，并在区间 [j+1,n-1] 二分查找第三个元素。由于结果不能重复，需要使用集合保存结果。

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Set<List<Integer>> res = new HashSet<>();
        //将数组排序
        Arrays.sort(nums);
        int n = nums.length;
        //枚举三元组前2个元素
        for(int i=0;i<n-2;i++){
            for(int j=i+1;j<n-1;j++){
                //二分查找第3个元素
                int k = bs(nums,j+1,n-1,-(nums[i]+nums[j]));
                if(k!=-1) res.add(new ArrayList<>(Arrays.asList(new Integer[]{nums[i],nums[j],nums[k]})));
            }
        }
        return new ArrayList<List<Integer>>(res);
    }

    int bs(int[] nums,int l,int r,int target){
        while(l<=r){
            int mid = (l+r)>>1;
            if(nums[mid] == target) return mid;
            else if(nums[mid] > target) r = mid-1;
            else l = mid+1;
        }
        return -1;
    }
}
```

思路二：

直接转换为nSum问题，遍历n元组开始元素的位置i, 递归求目标值为 target-nums[i] 的 (n-1) 元组，当n=2时，直接使用双指针求出结果即可。

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        // 将数组排序
        Arrays.sort(nums);
        return nSum(nums,3,0,0);
    }

    private List<List<Integer>> nSum(int[] nums,int n,int start,int target){
        int sz = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        // 至少是 2Sum，且搜索区间大小sz-start不小于n
        if(n<2 || sz-start<n) return res;
        // base case
        if(n==2){
            int lo = start,hi = sz-1;
            while(lo < hi){
                int left = nums[lo],right = nums[hi];
                int sum = left + right;
                if(sum < target){ //和较小，右移左指针
                    while(lo < hi && nums[lo] == left) lo++;
                }else if(sum > target){ //和较大，左移右指针
                    while(lo < hi && nums[hi] == right) hi--;
                }else{
                    //记录二元组
                    res.add(new ArrayList(Arrays.asList(new Integer[]{left,right})));
                    //更新左右指针的位置
                    while(lo < hi && nums[lo] == left) lo++;
                    while(lo < hi && nums[hi] == right) hi--;
                }
            }
        }else{
            // 遍历元组的开始位置
            for(int i=start;i<sz;i++){
                // 递归计算子结果
                List<List<Integer>> sub = nSum(nums,n-1,i+1,target-nums[i]);
                // 将开始元素添加到子结果中
                for(List<Integer> list:sub){
                    list.add(nums[i]);
                    res.add(list);
                }
                // 跳过重复元素
                while(i < sz-1 && nums[i] == nums[i+1]) i++;
            }
        }
        
        return res;
    }
}
```

[42. 接雨水 - 力扣（LeetCode）](https://leetcode.cn/problems/trapping-rain-water/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

每个柱子可以接到的雨水为当前柱子左边柱子的最大高度和右边柱子的最大高度的较小值减去当前柱子的高度，即：$water[i] = min(max(height[0..i],height[i..n-1]))-height[i]$，因此，直接计算出每个柱子左右两边的最大柱子高度lmax[i], rmax[i]，最后计算每个柱子可以接的雨水即可。

```java
class Solution {
    public int trap(int[] height) {
        int n=height.length;
        // lmax[i]: [0..i]区间的柱子高度的最大值
        int[] lmax = new int[n];
        // rmax[i]: [i..n-1]区间的柱子高度的最大值
        int[] rmax = new int[n];
        lmax[0] = height[0]; 
        rmax[n-1] = height[n-1];
        for(int i=1;i<n;i++){
            lmax[i] = Math.max(lmax[i-1],height[i]);
        }
        for(int i=n-2;i>=0;i--){
            rmax[i] = Math.max(rmax[i+1],height[i]);
        }
        int res = 0;
        // 计算每个柱子上可以接的雨水
        for(int i=0;i<n;i++){
            res += Math.min(lmax[i], rmax[i]) - height[i];
        }
        return res;
    }
}
```

思路二：

根据每个柱子接雨水的高度公式可以看出，当前柱子能接的水量是由左右最高柱子高度的较小值决定的，因此可以使用双指针left, right，在计算lmax[0..left]和rmax[right..n-1]的同时，计算每个柱子可以接的雨水：

- 若left < right：此时rmax[right..n-1] < rmax[left..n-1]，能接的雨水由lmax决定，res+=lmax-heigth[left]
- 若left > right：此时lmax[0..left] < lmax[0..right]，能接的雨水由rmax决定，res+=rmax-heigth[right]

```java
class Solution {
    public int trap(int[] height) {
        int n=height.length;
        int lmax=0,rmax=0;
        int left=0,right=n-1;
        int res=0;
        // 注意此处可以为<，因为最终height[left]=height[right]=maxHeight
        while(left<=right){
            lmax=Math.max(lmax,height[left]);
            rmax=Math.max(rmax,height[right]);
            if(lmax<rmax){
                res+=lmax-height[left];
                left++;
            }else{
                res+=rmax-height[right];
                right--;
            }
        }
        return res;
    }
}
```

## 滑动窗口

[3. 无重复字符的最长子串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-substring-without-repeating-characters/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        // 定义窗口
        Map<Character, Integer> window = new HashMap<>();
        
        int left = 0, right = 0;
        int res = 0;
        while(right < s.length()){
            char c = s.charAt(right++);
            // 进行窗口内数据的一系列更新
            window.put(c, window.getOrDefault(c, 0) + 1);
            // 判断左侧窗口是否要收缩
            while(window.get(c) > 1){
                char d = s.charAt(left++);
                // 进行窗口内数据的一系列更新
                window.put(d, window.get(d) - 1);
            }
            // 更新结果
            res = Math.max(res, right-left);
        }
        return res;
    }
}
```

[438. 找到字符串中所有字母异位词 - 力扣（LeetCode）](https://leetcode.cn/problems/find-all-anagrams-in-a-string/?envType=study-plan-v2&envId=top-100-liked)

思路：直接使用滑动窗口模板即可，使用need保存模式串p中所有字符，每次将一个字符添加到窗口中，当窗口中字符数量大于模式串p的长度时需要收缩窗口，使用count记录此时满足数量要求的字符数。收缩后窗口的长度即与模式串p的长度相等，如果满足要求的字符数等于need中的字符数，则找到了一个合法结果。

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        Map<Character, Integer> window = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        for(char c:p.toCharArray()) need.put(c, need.getOrDefault(c, 0) + 1);

        // 记录窗口中满足数量要求的字符数
        int count = 0;
        int left = 0, right = 0;
        // 记录结果
        List<Integer> res = new ArrayList<>();
        while(right < s.length()){
            char c = s.charAt(right++);
            // 进行窗口内的一些列更新
            if(need.containsKey(c)){
                window.put(c, window.getOrDefault(c, 0) + 1);
                if(window.get(c).equals(need.get(c))) count++;
            }
            // 判断左侧窗口是否要收缩
            while(right - left > p.length()){
                char d = s.charAt(left++);
                // 进行窗口内的一些列更新
                if(need.containsKey(d)){
                    if(window.get(d).equals(need.get(d))) count--;
                    window.put(d, window.get(d) - 1);
                }
            }
            // 满足数量要求的字符与模式串中相等，记录结果
            if(count == need.size()) res.add(left);
        }
        return res;
    }
}
```

## 子串

[560. 和为 K 的子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/subarray-sum-equals-k/?envType=study-plan-v2&envId=top-100-liked)

思路一：

题目即求元素之和为k的子数组，设子数组所在的区间为区间为$[i, j]$，则可以枚举每一个区间，统计满足条件的即可。可以现枚举区间的结束位置$j$, 再枚举开始位置$i$，开始位置可以反向遍历，边遍历边计算元素和。

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        int res = 0;
        //枚举结束位置
        for(int i=0;i<n;i++){
            int sum = 0;
            //枚举开始位置
            for(int j=i;j>=0;j--){
                sum += nums[j];
                if(sum == k) res++;
            }
        }
        return res;
    }
}
```

思路二：

- 可以先计算数组的前缀和，区间$[i,j]$的子数组元素之和即为$k=pre[j]-pre[i-1]$，则对于结束位置$j$，需要找到满足$pre[i-1]=pre[j]-k$的所有$i$。
- 可以使用哈希表map保存当前枚举结束位置$j$左侧各前缀和的出现次数，则当结束位置为$j$时满足条件的子数组个数为$map[pre[j]-k]$，累加到结果即可。
- 由于需要知道当前枚举结束位置前面特定前缀和的出现次数，需要边遍历边计算前缀和，即使用变量pre保存当前前缀和$nums[0..j]$，同时将前缀和保存在map映射中。

```java
class Solution {
    // pre[i]-pre[j-1] = k
    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        // map[x]=n表示前缀和数组中当前值为x的元素数量为n
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        // 记录当前的前缀和
        int pre = 0;
        // 初始化（前缀和为0的元素个数为1）
        map.put(0, 1);
        for(int i=0;i<n;i++){
            pre += nums[i];
            if(map.containsKey(pre - k)){
                res += map.get(pre - k);
            }
            map.put(pre, map.getOrDefault(pre, 0) + 1);
        }
        return res;
    }
}
```

[239. 滑动窗口最大值 - 力扣（LeetCode）](https://leetcode.cn/problems/sliding-window-maximum/?envType=study-plan-v2&envId=top-100-liked)

思路：

题目要求固定长度的窗口中的最大值，可以使用单调队列实现。这里使用LinkedList实现单调递减队列：

- 在添加元素时，将队尾比自身小的元素都弹出
- 在弹出一个元素时，若队首元素等于该元素则弹出，否则说明该元素在添加元素时已经被弹出，不做处理
- 在求窗口中的最大值时直接取队首元素即可

```java
class Solution {
    // 单调队列
    static class MonotonicQueue{
        LinkedList<Integer> maxq;
        public MonotonicQueue(){
            this.maxq=new LinkedList<>();
        }

        void push(int x){
            // 将前面小于x的元素弹出
            while(!maxq.isEmpty()&&maxq.getLast()<x) maxq.pollLast();
            maxq.addLast(x);
        }

        void pop(int x){
            // 当队头元素为x时，删除
            if(maxq.getFirst()==x) maxq.pollFirst();
        }

        int max(){
            // 队头元素为最大值
            return maxq.getFirst();
        }
    }
    public int[] maxSlidingWindow(int[] nums, int k) {
        MonotonicQueue window=new MonotonicQueue();
        int n=nums.length;
        int[] res=new int[n-k+1];
        for(int i=0;i<n;i++){
            if(i<k-1){
                // 先将前k-1个元素填入窗口
                window.push(nums[i]);
            }else{
                // 窗口开始向前滑动
                window.push(nums[i]);
                // 获取窗口最大元素
                res[i-k+1]=window.max();
                // 将离开窗口的元素删除
                window.pop(nums[i-k+1]);
            }
        }
        return res;
    }
}
```

[76. 最小覆盖子串 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-window-substring/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public String minWindow(String s, String t) {
        Map<Character, Integer> window = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        for(char c:t.toCharArray()) need.put(c, need.getOrDefault(c, 0)+1);

        int valid = 0;
        int left = 0,right = 0;
        // 记录最小覆盖字串的起始位置及长度
        int start = 0, len = Integer.MAX_VALUE;
        while(right < s.length()){
            char c = s.charAt(right++);
            // 进行窗口内数据的一系列更新
            if(need.containsKey(c)) {
                window.put(c, window.getOrDefault(c, 0)+1);
                if(window.get(c).equals(need.get(c))) valid++;
            }

            // 判断左侧窗口是否要收缩
            while(valid == need.size()){
                if(right-left < len){
                    start = left;
                    len = right - left;
                }
                char d = s.charAt(left++);
                // 进行窗口内数据的一系列更新
                if(need.containsKey(d)) {
                    if(window.get(d).equals(need.get(d))) valid--;
                    window.put(d, window.get(d)-1);
                }
            }
        }
        // 返回最小覆盖子串
        return len == Integer.MAX_VALUE?"":s.substring(start, start+len);
    }
}
```

## 普通数组

[53. 最大子数组和 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-subarray/?envType=study-plan-v2&envId=top-100-liked)

思路一：

使用分治算法，将数组分为两部分，最大的子数组和可能出现在左数组、右数组或左右两数组各取一部分，前2种情况可以直接递归求，第三种枚举子数组的开始位置和结束位置求，最终取三者的最大值即可，时间复杂度为O(n^3)。

```java
class Solution {
    int[] pre;
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        pre = new int[n+1];
        for(int i=0;i<n;i++) pre[i+1] = pre[i] + nums[i];
        return maxSubArray(nums, 0, n-1);
    }
    int maxSubArray(int[] nums, int lo, int hi){
        if(lo == hi) return nums[lo];
        int mid = (lo + hi) >> 1;
        int maxLeft = maxSubArray(nums, lo, mid);
        int maxRight = maxSubArray(nums, mid+1, hi);
        int res = Math.max(maxLeft, maxRight);
        for(int i=0;i<=mid;i++){
            for(int j=mid+1;j<=hi;j++){
                res = Math.max(res, pre[j+1] - pre[i]);
            }
        }
        return res;
    }
}
```

思路二：

使用动态规划，定义dp[i]为以nums[i]结尾的子数组的最大和，则对于nums[i]有两种选择：

- 拼接在前面的以nums[i-1]结尾的子数组，则dp[i] = dp[i-1] + nums[i]
- 独自成为一个子数组，则dp[i] = nums[i]

最终枚举子数组的结束位置，取最大值即可。

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        // dp[i]表示以nums[i]结尾的子数组的最大和
        int[] dp = new int[n];
        dp[0] = nums[0];
        for(int i=1;i<n;i++){
            dp[i] = Math.max(nums[i], dp[i-1] + nums[i]);
        }
        int res = Integer.MIN_VALUE;
        // 枚举每一个结尾
        for(int i=0;i<n;i++){
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}
```

[56. 合并区间 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-intervals/?envType=study-plan-v2&envId=top-100-liked)

思路：先将区间按照开始位置升序排序，对于每个区间[left, right]，设已合并的最后一个区间为[start, end]。若left <= end，则可以与该区间合并，更新最后一个合并区间的结束位置为Math.max(end, right)，否则需要作为一个新的合并区间。

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        // 将区间按开始位置升序排序
        Arrays.sort(intervals, (o1, o2) -> o1[0]-o2[0]);
        List<int[]> res = new ArrayList<>();
        // 加入第一个区间
        res.add(intervals[0]);
        for(int i = 1;i<intervals.length;i++){
            int[] cur = intervals[i];
            int[] last = res.get(res.size()-1);
            if(cur[0] <= last[1]){
                // 当前区间可以和最后一个区间合并，更新最后一个区间的结束位置
                last[1] = Math.max(last[1], cur[1]);
            }else{
                // 不能合并，将当前区间添加到res中
                res.add(cur);
            }
        }
        return res.toArray(new int[0][0]);
    }
}
```

[189. 轮转数组 - 力扣（LeetCode）](https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：使用额外的数组

使用一个额外的数组，将每个元素移到空数组对应位置，再将空数组复制回原数组即可。

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int n=nums.length;
        k%=n;
        int[] temp=new int[n];
        for(int i=0;i<n;i++) temp[(i+k)%n]=nums[i];
        for(int i=0;i<n;i++) nums[i]=temp[i];
    }
}
```

思路二：数组翻转

先将整个数组翻转，再分别将区间 [0, k-1] 和 [k, n-1] 的元素翻转，得到的结果即为轮转 k 步后的数组。

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int n=nums.length;
        k %= n;
        reverse(nums, 0, n-1);
        reverse(nums, 0, k-1);
        reverse(nums, k, n-1);
    }
    void reverse(int[] nums, int l, int r){
        while(l<r){
            int temp = nums[l];
            nums[l] = nums[r];
            nums[r] = temp;
        }
    }
}
```

思路三：环状替换

对于开始位置 start 的元素，需要被移动到 (start+k)%n 位置，使用 temp 保存被置换的元素，下一次将 temp 的值设置给下一个位置，依此类推，直到回到开始位置。但是此时还是有很多元素是没有遍历到的，因此需要从开始位置的下一个位置 start+1 开始继续替换，直到所有元素都被替换。

设每轮经过的圈数为x，每一轮遍历到的元素数量为 y，则有 xn = yk，显然 yk 是n和k的公倍数，又因为y尽可能小，故 yk 是n和k的最小公倍数，即  yk = nk/gcd(n, k)，n/y=gcd(n, k)，则遍历的轮数为gcd(n, k)。

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int n=nums.length;
        k %= n;
        int count = gcd(n, k);
        for(int start=0;start<count;start++){
            int i = start;
            int prev = nums[start];
            do{
                int j = (i+k)%n;
                int temp = nums[j];
                nums[j] = prev;
                prev = temp;
                i = j;
            }while(i != start);
        }
    }

    int gcd(int x, int y){
        return y == 0?x:gcd(y, x%y);
    }
}
```

[238. 除自身以外数组的乘积 - 力扣（LeetCode）](https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-100-liked)

思路一：

定义数组L, R分别表示 nums[i] 左侧和右侧的元素乘积，最终计算遍历数组计算每个元素除自身之外的元素乘积，时间复杂度为O(n^2)，空间复杂度为O(n)。

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        // L[i]:nums[i]左侧元素的乘积
        // R[i]:nums[i]右侧元素的乘积
        int[] L = new int[n], R = new int[n];
        L[0] = 1;
        for(int i=1;i<n;i++){
            L[i] = L[i-1] * nums[i-1];
        }
        R[n-1] = 1;
        for(int i=n-2;i>=0;i--){
            R[i] = R[i+1] * nums[i+1];
        }

        int[] res = new int[n];
        for(int i = 0;i<n;i++){
            res[i] = L[i] * R[i];
        }
        return res;
    }
}
```

思路二：

直接先使用 res 数组计算L数组，然后计算 R 数组，只是使用 R 变量保存，边遍历边计算结果，时间复杂度为O(n^2)，空间复杂度为O(1)。

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for(int i=1;i<n;i++){
            res[i] = res[i-1] * nums[i-1];
        }
        int R = 1;
        for(int i = n-2;i>=0;i--){
            R *= nums[i+1];
            res[i] *= R;
        }
        return res;
    }
}
```

[41. 缺失的第一个正数 - 力扣（LeetCode）](https://leetcode.cn/problems/first-missing-positive/?envType=study-plan-v2&envId=top-100-liked)

思路：

显然未出现的第一个正数的范围为 [1, n+1]，因此可以将范围 [1, n] 的元素 x 进行标记，最终找出区间 [1, n] 中未被标记的最小元素即为结果，如果都被标记则结果为 n+1。可以借助原始数组，先将不在范围 [1, n]的元素都置为 n+1，对于范围 [1, n] 的元素x，将 arr[x-1] 置为 -arr[x-1]，最终找出数组中非负数的最小位置即为结果。

```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        // 将[,0]范围的元素都设为n+1
        for(int i=0;i<n;i++){
            if(nums[i]<=0) nums[i] = n+1;
        }
        // 将[1,n]范围的元素x对应的位置x-1打上标记
        for(int i=0;i<n;i++){
            int num = Math.abs(nums[i]);
            if(num <= n){
                nums[num-1] = -Math.abs(nums[nums-1]);
            }
        }
        // 没有的打上标记的第一个元素即为结果
        for(int i=0;i<n;i++){
            if(nums[i] > 0) return i+1;
        }
        return n+1;
    }
}
```

## 矩阵

[73. 矩阵置零 - 力扣（LeetCode）](https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

先将值为0的元素坐标都保存到列表中，遍历列表中的坐标，将每个位置的横纵行元素都置为0，时间复杂度O(nm^2)。

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        // 值为0的元素坐标
        List<int[]> list = new ArrayList<>();
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(matrix[i][j] == 0) list.add(new int[]{i,j});
            }
        }
        // 将每个坐标对应的行列置为0
        for(int[] tp:list){
            for(int i = 0;i<m;i++) matrix[i][tp[1]] = 0;
            for(int j = 0;j<n;j++) matrix[tp[0]][j] = 0;
        }
    }
}
```

思路2：

使用首行和首列保存对应列和对应行是否需要置0，并使用 flag_col0, flag_row0 记录首行和首列需要置0，先将中间元素对应位置置0，再将首行和首列置0。

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        // 标记第一列和第一行是否需要置0
        boolean flag_col0 = false, flag_row0 = false;
        for(int i=0;i<m;i++){
            if(matrix[i][0] == 0){
                flag_col0 = true;
                break;
            } 
        }
        for(int j=0;j<n;j++){
            if(matrix[0][j] == 0){
                flag_row0 = true;
                break;
            }
        }

        // 标记需要置为0的行和列
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(matrix[i][j] == 0) {
                    matrix[0][j] = matrix[i][0] = 0;
                }
            }
        }

        // 遍历中间的元素并将对应元素置0
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(matrix[0][j] == 0 || matrix[i][0] == 0){
                    matrix[i][j] = 0;
                }
            }
        }

        // 置0首行和首列的
        if(flag_col0){
            for(int i=0;i<m;i++) matrix[i][0] = 0;
        }
        if(flag_row0){
            for(int j=0;j<n;j++) matrix[0][j] = 0;
        }
    }
}
```

[54. 螺旋矩阵 - 力扣（LeetCode）](https://leetcode.cn/problems/spiral-matrix/description/?envType=study-plan-v2&envId=top-100-liked)

思路：

记录当前未经过元素的上下左右边界，从矩阵左上角开始，依次向右下左上遍历，每遍历完一个方向，更新对应的边界值，当无法继续遍历时结束。

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        // 定义边界
        int upper = 0, lower = m-1, left = 0, right = n-1;

        List<Integer> res = new ArrayList<>();
        while(true){
            for(int j=left;j<=right;j++) res.add(matrix[upper][j]); //向右移动直到最右
            if(++upper>lower) break;
            for(int i=upper;i<=lower;i++) res.add(matrix[i][right]); //向下
            if(--right<left) break;
            for(int j=right;j>=left;j--) res.add(matrix[lower][j]); //向左
            if(--lower<upper) break;
            for(int i=lower;i>=upper;i--) res.add(matrix[i][left]); //向上
            if(++left>right) break;
        }

        return res;
    }
}
```

[48. 旋转图像 - 力扣（LeetCode）](https://leetcode.cn/problems/rotate-image/?envType=study-plan-v2&envId=top-100-liked)

思路：

顺时针旋转矩阵90度后，矩阵的行都变成了列。将矩阵按照对角线翻转后，可以发现结果矩阵和旋转后的矩阵是沿竖线对称的，因此将翻转后的矩阵每一行转置即可得到旋转矩阵。

```java
class Solution {
    public void rotate(int[][] matrix) {
        int n=matrix.length;
        // 沿对角线翻转矩阵
        for(int i=0;i<n;i++){
            for(int j=i;j<n;j++){
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }

        // 翻转矩阵的每一行
        for(int i=0;i<n;i++){
            for(int j=0;j<n/2;j++){
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n-1-j];
                matrix[i][n-1-j] = temp;
            }
        }
    }
}
```

[240. 搜索二维矩阵 II - 力扣（LeetCode）](https://leetcode.cn/problems/search-a-2d-matrix-ii/?envType=study-plan-v2&envId=top-100-liked)

思路一：二分查找

由于矩阵的每一行都是有序的，因此可以对每一行进行二分查找，找到了结果即可返回。

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        //二分搜索每一行
        for(int[] row:matrix){
            if(row[0]>target) return -1;
            int idx = bs(row, target);
            if(idx != -1) return true;
        }
        return false;
    }

    int bs(int[] arr, int target){
        int left = 0, right = arr.length-1;
        while(left <= right){
            int mid = (left + right) >> 1;
            if(arr[mid] == target) return mid;
            else if(arr[mid] > target) right = mid-1;
            else left = mid+1;
        }
        return -1;
    }
}
```

思路二：搜索

矩阵的每一行和每一列都是升序的，可以发现，在矩阵的右上角时，向左移动元素值会减小，向下移动时元素值会增大。因此可以从右上角开始搜索，若元素较大，向左移动，否则向下移动，直到找到目标值结束。

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        // 从右上角开始搜索
        int x = 0, y = n-1;
        while(x<m && y>=0){
            if(matrix[x][y] == target) return true;
            else if(matrix[x][y] > target) y--;
            else x++;
        }
        return false;
    }
}
```


