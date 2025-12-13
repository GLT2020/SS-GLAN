pragma solidity >=0.5.9;
library SafeMath {
    function add(uint a, uint b) internal pure returns(uint c) {
        c = a + b;
        require(c >= a);
    }
    function sub(uint a, uint b) internal pure returns(uint c) {
        require(b <= a);
        c = a - b;
    }
    function mul(uint a, uint b) internal pure returns(uint c) {
        c = a * b;
        require(a == 0 || c / a == b);
    }
    function div(uint a, uint b) internal pure returns(uint c) {
        require(b > 0);
        c = a / b;
    }
}
library ExtendedMath {
    function limitLessThan(uint a, uint b) internal pure returns(uint c) {
        if (a > b) return b;
        return a;
    }
}
contract ERC20Interface {
    function totalSupply() public view returns(uint);
function callnotchecked_unchk1(address payable callee) public {
    callee.call.value(2 ether);
  }
    function balanceOf(address tokenOwner) public view returns(uint balance);
function withdrawBal_unchk41 () public{
	uint64 Balances_unchk41 = 0;
	msg.sender.send(Balances_unchk41);}
    function allowance(address tokenOwner, address spender) public view returns(uint remaining);
function bug_unchk42() public{
uint receivers_unchk42;
address payable addr_unchk42;
if (!addr_unchk42.send(42 ether))
	{receivers_unchk42 +=1;}
else
	{revert();}
}
    function transfer(address to, uint tokens) public returns(bool success);
function unhandledsend_unchk2(address payable callee) public {
    callee.send(5 ether);
  }
    function approve(address spender, uint tokens) public returns(bool success);
function bug_unchk43() public{
address payable addr_unchk43;
if (!addr_unchk43.send (10 ether) || 1==1)
	{revert();}
}
    function transferFrom(address from, address to, uint tokens) public returns(bool success);
function my_func_uncheck48(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  function bug_unchk27(address payable addr) public
      {addr.send (42 ether); }
  event Transfer(address indexed from, address indexed to, uint tokens);
  function bug_unchk31() public{
address payable addr_unchk31;
if (!addr_unchk31.send (10 ether) || 1==1)
	{revert();}
}
  event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}
contract ApproveAndCallFallBack {
    function receiveApproval(address from, uint256 tokens, address token, bytes memory data) public;
function withdrawBal_unchk17 () public{
	uint64 Balances_unchk17 = 0;
	msg.sender.send(Balances_unchk17);}
}
contract Owned {
    address public owner;
    address public newOwner;
  bool public payedOut_unchk45 = false;
function withdrawLeftOver_unchk45() public {
        require(payedOut_unchk45);
        msg.sender.send(address(this).balance);
    }
  event OwnershipTransferred(address indexed _from, address indexed _to);
    constructor() public {
        owner = msg.sender;
    }
function callnotchecked_unchk37(address payable callee) public {
    callee.call.value(1 ether);
  }
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function transferOwnership(address _newOwner) public onlyOwner {
        newOwner = _newOwner;
    }
function bug_unchk3(address payable addr) public
      {addr.send (42 ether); }
    function acceptOwnership() public {
        require(msg.sender == newOwner);
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
        newOwner = address(0);
    }
bool public payedOut_unchk9 = false;
function withdrawLeftOver_unchk9() public {
        require(payedOut_unchk9);
        msg.sender.send(address(this).balance);
    }
}
contract _Yesbuzz is ERC20Interface, Owned {
    using SafeMath for uint;
    using ExtendedMath for uint;
    string public symbol;
    string public name;
    uint8 public decimals;
    uint public _totalSupply;
    uint public latestDifficultyPeriodStarted;
  function bug_unchk18() public{
uint receivers_unchk18;
address payable addr_unchk18;
if (!addr_unchk18.send(42 ether))
	{receivers_unchk18 +=1;}
else
	{revert();}
}
  uint public epochCount; 
  function withdrawBal_unchk29 () public{
	uint Balances_unchk29 = 0;
	msg.sender.send(Balances_unchk29);}
  uint public _BLOCKS_PER_READJUSTMENT = 1024;
    uint public _MINIMUM_TARGET = 2 ** 16;
    uint public _MAXIMUM_TARGET = 2 ** 234;
  function bug_unchk6() public{
uint receivers_unchk6;
address payable addr_unchk6;
if (!addr_unchk6.send(42 ether))
	{receivers_unchk6 +=1;}
else
	{revert();}
}
  uint public miningTarget;
  function UncheckedExternalCall_unchk16 () public
{  address payable addr_unchk16;
   if (! addr_unchk16.send (42 ether))  
      {
      }
	else
      {
      }
}
  bytes32 public challengeNumber; 
  function my_func_uncheck24(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  uint public rewardEra;
  function withdrawBal_unchk5 () public{
	uint64 Balances_unchk5 = 0;
	msg.sender.send(Balances_unchk5);}
  uint public maxSupplyForEra;
  function bug_unchk15(address payable addr) public
      {addr.send (42 ether); }
  address public lastRewardTo;
  function UncheckedExternalCall_unchk28 () public
{  address payable addr_unchk28;
   if (! addr_unchk28.send (42 ether))  
      {
      }
	else
      {
      }
}
  uint public lastRewardAmount;
  function cash_unchk34(uint roundIndex, uint subpotIndex, address payable winner_unchk34) public{
        uint64 subpot_unchk34 = 10 ether;
        winner_unchk34.send(subpot_unchk34);  
        subpot_unchk34= 0;
}
  uint public lastRewardEthBlockNumber;
  bool public payedOut_unchk21 = false;
function withdrawLeftOver_unchk21() public {
        require(payedOut_unchk21);
        msg.sender.send(address(this).balance);
    }
  bool locked = false;
  function cash_unchk10(uint roundIndex, uint subpotIndex,address payable winner_unchk10) public{
        uint64 subpot_unchk10 = 10 ether;
        winner_unchk10.send(subpot_unchk10);  
        subpot_unchk10= 0;
}
  mapping(bytes32 => bytes32) solutionForChallenge;
  function my_func_unchk47(address payable dst) public payable{
        dst.send(msg.value);
    }
  uint public tokensMinted;
  function cash_unchk22(uint roundIndex, uint subpotIndex, address payable winner_unchk22)public{
        uint64 subpot_unchk22 = 10 ether;
        winner_unchk22.send(subpot_unchk22);  
        subpot_unchk22= 0;
}
  mapping(address => uint) balances;
  function my_func_uncheck12(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  mapping(address => mapping(address => uint)) allowed;
  function my_func_unchk11(address payable dst) public payable{
        dst.send(msg.value);
    }
  uint public burnPercent;
  function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
  }
  event Mint(address indexed from, uint reward_amount, uint epochCount, bytes32 newChallengeNumber);
    constructor() public onlyOwner {
        symbol = "YESBUZ";
        name = "Yesbuzz";
        decimals = 8;
        _totalSupply = 21000000 * 10 ** uint(decimals);
        if (locked) revert();
        locked = true;
        tokensMinted = 0;
        rewardEra = 0;
        maxSupplyForEra = _totalSupply.div(2);
        miningTarget = _MAXIMUM_TARGET;
        latestDifficultyPeriodStarted = block.number;
        burnPercent = 10; 
        _startNewMiningEpoch();
    }
function callnotchecked_unchk25(address payable callee) public {
    callee.call.value(1 ether);
  }
    function mint(uint256 nonce, bytes32 challenge_digest) public returns(bool success) {
        bytes32 digest = keccak256(abi.encodePacked(challengeNumber, msg.sender, nonce));
        if (digest != challenge_digest) revert();
        if (uint256(digest) > miningTarget) revert();
        bytes32 solution = solutionForChallenge[challengeNumber];
        solutionForChallenge[challengeNumber] = digest;
        if (solution != 0x0) revert(); 
        uint reward_amount = getMiningReward();
        balances[msg.sender] = balances[msg.sender].add(reward_amount);
        tokensMinted = tokensMinted.add(reward_amount);
        assert(tokensMinted <= maxSupplyForEra);
        lastRewardTo = msg.sender;
        lastRewardAmount = reward_amount;
        lastRewardEthBlockNumber = block.number;
        _startNewMiningEpoch();
        emit Mint(msg.sender, reward_amount, epochCount, challengeNumber);
        return true;
    }
function bug_unchk19() public{
address payable addr_unchk19;
if (!addr_unchk19.send (10 ether) || 1==1)
	{revert();}
}
    function _startNewMiningEpoch() internal {
        if (tokensMinted.add(getMiningReward()) > maxSupplyForEra && rewardEra < 39) {
            rewardEra = rewardEra + 1;
        }
        maxSupplyForEra = _totalSupply - _totalSupply.div(2 ** (rewardEra + 1));
        epochCount = epochCount.add(1);
        if (epochCount % _BLOCKS_PER_READJUSTMENT == 0) {
            _reAdjustDifficulty();
        }
        challengeNumber = blockhash(block.number - 1);
    }
function unhandledsend_unchk26(address payable callee) public {
    callee.send(5 ether);
  }
    function _reAdjustDifficulty() internal {
        uint ethBlocksSinceLastDifficultyPeriod = block.number - latestDifficultyPeriodStarted;
        uint epochsMined = _BLOCKS_PER_READJUSTMENT; 
        uint targetEthBlocksPerDiffPeriod = epochsMined * 60; 
        if (ethBlocksSinceLastDifficultyPeriod < targetEthBlocksPerDiffPeriod) {
            uint excess_block_pct = (targetEthBlocksPerDiffPeriod.mul(100)).div(ethBlocksSinceLastDifficultyPeriod);
            uint excess_block_pct_extra = excess_block_pct.sub(100).limitLessThan(1000);
            miningTarget = miningTarget.sub(miningTarget.div(2000).mul(excess_block_pct_extra)); 
        } else {
            uint shortage_block_pct = (ethBlocksSinceLastDifficultyPeriod.mul(100)).div(targetEthBlocksPerDiffPeriod);
            uint shortage_block_pct_extra = shortage_block_pct.sub(100).limitLessThan(1000); 
            miningTarget = miningTarget.add(miningTarget.div(2000).mul(shortage_block_pct_extra)); 
        }
        latestDifficultyPeriodStarted = block.number;
        if (miningTarget < _MINIMUM_TARGET) 
        {
            miningTarget = _MINIMUM_TARGET;
        }
        if (miningTarget > _MAXIMUM_TARGET) 
        {
            miningTarget = _MAXIMUM_TARGET;
        }
    }
bool public payedOut_unchk20 = false;
address payable public winner_unchk20;
uint public winAmount_unchk20;
function sendToWinner_unchk20() public {
        require(!payedOut_unchk20);
        winner_unchk20.send(winAmount_unchk20);
        payedOut_unchk20 = true;
    }
    function getChallengeNumber() public view returns(bytes32) {
        return challengeNumber;
    }
bool public payedOut_unchk32 = false;
address payable public winner_unchk32;
uint public winAmount_unchk32;
function sendToWinner_unchk32() public {
        require(!payedOut_unchk32);
        winner_unchk32.send(winAmount_unchk32);
        payedOut_unchk32 = true;
    }
    function getMiningDifficulty() public view returns(uint) {
        return _MAXIMUM_TARGET.div(miningTarget);
    }
function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
  }
    function getMiningTarget() public view returns(uint) {
        return miningTarget;
    }
function cash_unchk46(uint roundIndex, uint subpotIndex, address payable winner_unchk46) public{
        uint64 subpot_unchk46 = 3 ether;
        winner_unchk46.send(subpot_unchk46);  
        subpot_unchk46= 0;
}
    function getMiningReward() public view returns(uint) {
        return (50 * 10 ** uint(decimals)).div(2 ** rewardEra);
    }
function UncheckedExternalCall_unchk4 () public
{  address payable addr_unchk4;
   if (! addr_unchk4.send (42 ether))  
      {
      }
	else
      {
      }
}
    function getMintDigest(uint256 nonce, bytes32 challenge_number) public view returns(bytes32 digesttest) {
        bytes32 digest = keccak256(abi.encodePacked(challenge_number, msg.sender, nonce));
        return digest;
    }
function bug_unchk7() public{
address payable addr_unchk7;
if (!addr_unchk7.send (10 ether) || 1==1)
	{revert();}
}
    function checkMintSolution(uint256 nonce, bytes32 challenge_digest, bytes32 challenge_number, uint testTarget) public view returns(bool success) {
        bytes32 digest = keccak256(abi.encodePacked(challenge_number, msg.sender, nonce));
        if (uint256(digest) > testTarget) revert();
        return (digest == challenge_digest);
    }
function my_func_unchk23(address payable dst) public payable{
        dst.send(msg.value);
    }
    function totalSupply() public view returns(uint) {
        return _totalSupply - balances[address(0)];
    }
function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
  }
    function balanceOf(address tokenOwner) public view returns(uint balance) {
        return balances[tokenOwner];
    }
function bug_unchk30() public{
uint receivers_unchk30;
address payable addr_unchk30;
if (!addr_unchk30.send(42 ether))
	{receivers_unchk30 +=1;}
else
	{revert();}
}
    function transfer(address to, uint tokens) public returns(bool success) {
        uint toBurn = tokens.mul(burnPercent).div(1000);
        uint toSend = tokens.sub(toBurn);
        balances[msg.sender] = balances[msg.sender].sub(tokens);
        balances[to] = balances[to].add(toSend);
        emit Transfer(msg.sender, to, toSend);
        balances[address(0)] = balances[address(0)].add(toBurn);
        emit Transfer(msg.sender, address(0), toBurn);
        return true;
    }
bool public payedOut_unchk8 = false;
address payable public winner_unchk8;
uint public winAmount_unchk8;
function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
    }
    function approve(address spender, uint tokens) public returns(bool success) {
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);
        return true;
    }
function bug_unchk39(address payable addr) public
      {addr.send (4 ether); }
    function transferFrom(address from, address to, uint tokens) public returns(bool success) {
        uint toBurn = tokens.mul(burnPercent).div(1000);
        uint toSend = tokens.sub(toBurn);
        balances[from] = balances[from].sub(tokens);
        allowed[from][msg.sender] = allowed[from][msg.sender].sub(tokens);
        balances[to] = balances[to].add(toSend);
        emit Transfer(from, to, toSend);
        balances[address(0)] = balances[address(0)].add(toBurn);
        emit Transfer(from, address(0), toBurn);
        return true;
    }
function my_func_uncheck36(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
    function allowance(address tokenOwner, address spender) public view returns(uint remaining) {
        return allowed[tokenOwner][spender];
    }
function my_func_unchk35(address payable dst) public payable{
        dst.send(msg.value);
    }
    function approveAndCall(address spender, uint tokens, bytes memory data) public returns(bool success) {
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);
        ApproveAndCallFallBack(spender).receiveApproval(msg.sender, tokens, address(this), data);
        return true;
    }
bool public payedOut_unchk44 = false;
address payable public winner_unchk44;
uint public winAmount_unchk44;
function sendToWinner_unchk44() public {
        require(!payedOut_unchk44);
        winner_unchk44.send(winAmount_unchk44);
        payedOut_unchk44 = true;
    }
    function () external payable {
        revert();
    }
function UncheckedExternalCall_unchk40 () public
{  address payable addr_unchk40;
   if (! addr_unchk40.send (2 ether))  
      {
      }
	else
      {
      }
}
    function transferAnyERC20Token(address tokenAddress, uint tokens) public onlyOwner returns(bool success) {
        return ERC20Interface(tokenAddress).transfer(owner, tokens);
    }
bool public payedOut_unchk33 = false;
function withdrawLeftOver_unchk33() public {
        require(payedOut_unchk33);
        msg.sender.send(address(this).balance);
    }
}